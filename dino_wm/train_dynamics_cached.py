"""Fast warm-start dynamics fine-tune on CACHED model-step latents.

Drop-in replacement for the dataloader-bound `train.py env=pusht_multicolor` predictor
retrain: trains VWorldModel.forward_latent on scripts/cache_dynamics_latents.py output,
skipping the per-step mp4 decode + frozen-DINOv2 forward (~16h/epoch -> minutes/epoch).
The encoder stays frozen (it is never even invoked); only predictor + action/proprio
encoders update, with train.py's optimizer setup (AdamW predictor_lr / AdamW
action_encoder_lr over action+proprio). Saves a ckpt in train.py's exact format so
plan_multicolor.py / dynamics_check.py load it unchanged.

Run (box):
    python train_dynamics_cached.py \
      --warm_start outputs/2026-06-09/23-16-24 --warm_epoch 9 \
      --dyn_dir $DATASET_DIR/pusht_multicolor_10k/dyn_latents \
      --out outputs/2026-06-11/retrain10k_cached --epochs 8 --batch_size 32
"""
import argparse
import itertools
import json
import os
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets.dyn_latent_dset import DynLatentSliceDataset, StrideOneLatentDataset, CleanDynLatentDataset
from plan import load_model
from utils import move_to_device


def evaluate(model, loader, device):
    model.eval()
    tot = torch.zeros((), device=device); n = 0
    with torch.no_grad():
        for obs, act, _ in loader:
            obs = move_to_device(obs, device); act = act.to(device)
            _, _, _, loss, _ = model.forward_latent(obs["visual"], obs["proprio"], act)
            tot += loss.detach() * act.shape[0]; n += act.shape[0]
    return (tot / max(1, n)).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warm_start", required=True, help="dir with hydra.yaml + checkpoints/")
    ap.add_argument("--warm_epoch", default="latest")
    ap.add_argument("--dyn_dir", required=True, help="<...>/dyn_latents (has train/ val/)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--max_traj", type=int, default=None, help="train on first k trajs (scaling curve)")
    ap.add_argument("--stride_one", action="store_true", help="use per-frame StrideOneLatentDataset (full phase aug; needs dyn_latents_pf)")
    ap.add_argument("--clean", action="store_true", help="use variable-length CleanDynLatentDataset (pusht_noise clean retrain; dyn_latents_clean)")
    ap.add_argument("--lr", type=float, default=None, help="override predictor & action/proprio LR (default: cfg predictor_lr=5e-4; use ~1e-4 for a LIGHT warm-start adaptation)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model_cfg = OmegaConf.load(os.path.join(args.warm_start, "hydra.yaml"))
    ckpt = Path(args.warm_start) / "checkpoints" / f"model_{args.warm_epoch}.pth"
    model = load_model(ckpt, model_cfg, model_cfg.num_action_repeat, device=device)
    model.to(device)
    start_epoch = int(torch.load(ckpt, map_location="cpu").get("epoch", 0))
    # forward_latent is predictor-only and REQUIRES model.decoder is None to run, but the
    # copied hydra.yaml keeps has_decoder=true -> load_model would raise on reload if the
    # saved ckpt had no decoder. So stash the (untrained) VQ-VAE decoder and re-save it
    # verbatim in every checkpoint, while nulling it on the live model for training.
    saved_decoder = model.decoder
    model.decoder = None
    for p in model.encoder.parameters():
        p.requires_grad_(False)

    num_frames = model_cfg.num_hist + model_cfg.num_pred
    DS = (CleanDynLatentDataset if args.clean
          else StrideOneLatentDataset if args.stride_one
          else DynLatentSliceDataset)
    tr = DS(Path(args.dyn_dir) / "train", num_frames, max_traj=args.max_traj)
    va = DS(Path(args.dyn_dir) / "val", num_frames)
    tl = DataLoader(tr, batch_size=args.batch_size, shuffle=True, drop_last=True,
                    num_workers=args.num_workers, pin_memory=True)
    vl = DataLoader(va, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True)

    pred_lr = args.lr if args.lr is not None else model_cfg.training.predictor_lr
    act_lr = args.lr if args.lr is not None else model_cfg.training.action_encoder_lr
    print(f"[lr] predictor_lr={pred_lr} action_encoder_lr={act_lr}"
          + ("  (overridden for light warm-start)" if args.lr is not None else ""))
    predictor_opt = torch.optim.AdamW(model.predictor.parameters(), lr=pred_lr)
    act_opt = torch.optim.AdamW(itertools.chain(model.action_encoder.parameters(),
                                                model.proprio_encoder.parameters()),
                                lr=act_lr)

    out = Path(args.out); (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    OmegaConf.save(model_cfg, out / "hydra.yaml")  # so plan/dynamics_check resolve it
    hist = []
    val0 = evaluate(model, vl, device)
    print(f"[warm-start ep{start_epoch}] val_loss(cached) = {val0:.4f}")

    for e in range(1, args.epochs + 1):
        epoch = start_epoch + e
        model.train()
        run = torch.zeros((), device=device); n = 0
        for obs, act, _ in tqdm(tl, desc=f"Epoch {epoch} (cached)"):
            obs = move_to_device(obs, device); act = act.to(device)
            _, _, _, loss, _ = model.forward_latent(obs["visual"], obs["proprio"], act)
            predictor_opt.zero_grad(); act_opt.zero_grad()
            loss.backward()
            predictor_opt.step(); act_opt.step()
            run += loss.detach() * act.shape[0]; n += act.shape[0]   # on-GPU accum, no per-step sync
        train_loss = (run / max(1, n)).item()
        val_loss = evaluate(model, vl, device)
        print(f"Epoch {epoch}  Training loss: {train_loss:.4f}  Validation loss: {val_loss:.4f}")
        hist.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        ck = {"predictor": model.predictor, "action_encoder": model.action_encoder,
              "proprio_encoder": model.proprio_encoder, "decoder": saved_decoder,
              "epoch": epoch,
              "predictor_optimizer": predictor_opt, "action_encoder_optimizer": act_opt}
        torch.save(ck, out / "checkpoints" / "model_latest.pth")
        torch.save(ck, out / "checkpoints" / f"model_{epoch}.pth")
        json.dump(hist, open(out / "train_history.json", "w"), indent=2)
    print(f"Done -> {out}")


if __name__ == "__main__":
    main()

"""
Phase 0.3 grounding probe: is the target COLOR linearly decodable from DINOv2
patch features at the decal locations?

Trains a linear probe (384 -> n_colors + background) on per-patch features of the
cached START latents, with patches labeled by which colored target-outline they
overlap (block/agent-occluded patches dropped). High held-out accuracy => text
can be grounded into the patch grid. If a color is poorly decodable (esp. blue vs
the RoyalBlue pusher), make decals thicker/more saturated and re-cache.

    cd dino_wm
    /Users/Tom/miniforge3/envs/dino_wm_dev/bin/python analysis/grounding_probe.py \
        --data_path data/pusht_multicolor --split train
"""
import os
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# allow `python analysis/grounding_probe.py` from the repo root
import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from analysis.probe_common import (
    load_probe_data, build_grounding_dataset, episode_split, color_id_map)


def read_n_targets(data_path, labels):
    man = Path(data_path) / "split_manifest.json"
    if man.exists():
        return json.load(open(man)).get("n_targets", len(labels[0]["target_colors"]))
    return len(labels[0]["target_colors"])


def train_linear_probe(Xtr, ytr, Xte, n_classes, epochs=400, lr=1e-2, wd=1e-4, device="cpu"):
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6
    Xtr, Xte = ((Xtr - mu) / sd).to(device), ((Xte - mu) / sd).to(device)
    ytr = ytr.to(device)
    clf = nn.Linear(Xtr.shape[1], n_classes).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.CrossEntropyLoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(clf(Xtr), ytr)
        loss.backward()
        opt.step()
    with torch.no_grad():
        return clf(Xte).argmax(1).cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default=os.path.join(os.environ.get("DATASET_DIR", "data"), "pusht_multicolor"))
    ap.add_argument("--latent_dir", default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--outline_thickness", type=int, default=7)
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--out", default="analysis_outputs/grounding_probe.json")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    latent_dir = args.latent_dir or os.path.join(args.data_path, "latents")

    data = load_probe_data(args.data_path, latent_dir, args.split)
    n_targets = read_n_targets(args.data_path, data["labels"])
    names = [n for n, _ in color_id_map(n_targets).items()]

    X, y, ep = build_grounding_dataset(data, n_targets, args.outline_thickness)
    n_eps = len(data["labels"])
    tr, te = episode_split(ep, n_eps, frac=args.test_frac)
    print(f"patches: {len(y)} total ({int((y < n_targets).sum())} colored); "
          f"train {tr.sum()} / test {te.sum()} (by episode)")

    yhat = train_linear_probe(X[tr], y[tr], X[te], n_targets + 1, device=args.device)
    yte = y[te]

    color_mask = yte < n_targets
    overall_acc = float((yhat == yte).float().mean())
    color_acc = float((yhat[color_mask] == yte[color_mask]).float().mean()) if color_mask.any() else float("nan")

    per_color = {}
    for cid, name in enumerate(names):
        m = yte == cid
        per_color[name] = float((yhat[m] == cid).float().mean()) if m.any() else None
    conf = np.zeros((n_targets + 1, n_targets + 1), int)
    for t, p in zip(yte.tolist(), yhat.tolist()):
        conf[t, p] += 1

    report = {
        "split": args.split, "n_targets": n_targets,
        "chance_color": 1.0 / n_targets,
        "overall_acc_incl_bg": overall_acc,
        "color_only_acc": color_acc,
        "per_color_recall": per_color,
        "confusion_rows_true_cols_pred": conf.tolist(),
        "classes": names + ["background"],
        "grounding_feasible": bool(color_acc > 0.7),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=2)
    print(json.dumps({k: report[k] for k in
                      ["color_only_acc", "chance_color", "per_color_recall", "grounding_feasible"]}, indent=2))
    print(f"\nFull report -> {args.out}")
    if not report["grounding_feasible"]:
        print("WARN: color not cleanly decodable -> thicken/saturate decals and re-cache.")


if __name__ == "__main__":
    main()

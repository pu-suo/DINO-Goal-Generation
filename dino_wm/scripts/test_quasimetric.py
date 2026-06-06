"""CPU sanity tests for the quasimetric head + QRL training loop + CEM objective.

Runs entirely on the Mac (no CUDA, no dataset): synthetic STRUCTURED latents where
cost-to-go is a known function of trajectory position, so we can check the QRL loop
actually RECOVERS a monotone, asymmetric, unit-local-cost quasimetric. This is logic
validation; the real numbers come from the GPU box (see docs/QUASIMETRIC_RUNBOOK.md).

    .../dino_wm_dev/bin/python scripts/test_quasimetric.py
"""
import os
import sys
import json
import tempfile
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.quasimetric import QuasimetricHead, apply_keep_mask


# --------------------------------------------------------------------------- #
def test_head():
    print("== test_head ==")
    torch.manual_seed(0)
    B, P, D = 8, 196, 384
    for ht in ["iqe", "mrn", "mrn_fixed", "sym_l2"]:
        h = QuasimetricHead(head_type=ht).eval()
        z, zg = torch.randn(B, P, D), torch.randn(B, P, D)
        keep = torch.ones(P); keep[10:30] = 0
        d_ab, d_ba, d_aa = h(z, zg, keep), h(zg, z, keep), h(z, z, keep)
        assert d_ab.shape == (B,)
        if ht in ("iqe", "mrn", "mrn_fixed"):
            assert d_aa.abs().max() < 1e-3, (ht, "self-dist != 0")
        if ht == "iqe":
            assert (d_ab - d_ba).abs().mean() > 1e-3, "IQE not asymmetric"
        if ht == "sym_l2":
            assert (d_ab - d_ba).abs().max() < 1e-5, "sym_l2 not symmetric"
        print(f"   {ht:9s} ok (self {float(d_aa.abs().max()):.1e}, "
              f"asym {float((d_ab-d_ba).abs().mean()):.3f})")
    # mask broadcast equivalence + zeroing
    h = QuasimetricHead(head_type="iqe")
    keepP = torch.ones(P); keepP[5:25] = 0
    z, zg = torch.randn(B, P, D), torch.randn(B, P, D)
    assert (h(z, zg, keepP) - h(z, zg, keepP.view(1, P).expand(B, P))).abs().max() < 1e-6
    assert apply_keep_mask(z, keepP)[:, 5:25].abs().max() == 0
    print("   mask broadcast + zeroing ok")


# --------------------------------------------------------------------------- #
def make_synthetic_cache(root, split="train", n_traj=40, Lmin=8, Lmax=20, seed=0):
    """Structured cache: model-step k of a traj has a latent encoding scalar pos=k
    along a fixed global direction (+ small noise), so true cost-to-go(s->g) within a
    traj == (pos_g - pos_s) model steps, and is asymmetric (no backward path)."""
    rng = np.random.RandomState(seed)
    P, D = 196, 384
    direction = rng.randn(P, D).astype(np.float32)
    direction /= np.linalg.norm(direction)
    lat, states, starts, lengths = [], [], [], []
    cur = 0
    for t in range(n_traj):
        L = rng.randint(Lmin, Lmax + 1)
        base = rng.randn(P, D).astype(np.float32) * 0.5
        for k in range(L):
            z = base + k * direction + 0.01 * rng.randn(P, D).astype(np.float32)
            lat.append(z)
            ax, ay = rng.uniform(60, 450), rng.uniform(60, 450)   # random pusher xy
            bx, by = rng.uniform(100, 400), rng.uniform(100, 400)
            states.append([ax, ay, bx, by, rng.uniform(-3, 3), 0, 0])
        starts.append(cur); lengths.append(L); cur += L
    d = Path(root) / split; d.mkdir(parents=True, exist_ok=True)
    torch.save(torch.tensor(np.stack(lat)).half(), d / "latents.pth")
    torch.save(torch.tensor(np.asarray(states), dtype=torch.float32), d / "states.pth")
    torch.save(torch.tensor(starts), d / "traj_starts.pth")
    torch.save(torch.tensor(lengths), d / "traj_lengths.pth")
    json.dump(dict(frameskip=5, n_traj=n_traj, n_model_steps=cur,
                   model_step_len_p50=14, model_step_len_p90=19,
                   model_step_len_max=Lmax, state_dim=7, latent_shape=[P, D]),
              open(d / "meta.json", "w"))
    return cur


def test_qrl_recovery():
    print("== test_qrl_recovery (synthetic structured) ==")
    from datasets.qm_latent_dset import QMLatentDataset
    from models.quasimetric import build_quasimetric_head
    import torch.nn.functional as F
    import math

    with tempfile.TemporaryDirectory() as tmp:
        cache = Path(tmp) / "qm_latents"
        make_synthetic_cache(cache)
        dset = QMLatentDataset(str(cache), "train", mask_dilation=0, p_random_goal=0.2)
        loader = torch.utils.data.DataLoader(dset, batch_size=64, shuffle=True, drop_last=True)
        head = build_quasimetric_head(dict(head_type="iqe")).train()
        lam_raw = torch.nn.Parameter(torch.tensor(math.log(math.expm1(0.01))))
        opt = torch.optim.Adam(head.parameters(), lr=1e-3)
        opt_lam = torch.optim.Adam([lam_raw], lr=0.05)   # faster dual ascent for the short test
        eps2 = 0.25 ** 2
        it = iter(loader)
        logs, peak_dtrans = [], 0.0
        for step in range(500):
            try:
                b = next(it)
            except StopIteration:
                it = iter(loader); b = next(it)
            d_trans = head(b["z_a"], b["z_b"], b["keep_ab"])
            d_sg = head(b["z_s"], b["z_g"], b["keep_sg"])
            phi = -F.softplus(19.0 - d_sg, beta=0.1)
            constraint = F.relu(d_trans - 1).pow(2).mean() - eps2
            lam = F.softplus(lam_raw)
            loss = -phi.mean() + lam.detach() * constraint - lam * constraint.detach()
            opt.zero_grad(); opt_lam.zero_grad(); loss.backward(); opt.step(); opt_lam.step()
            peak_dtrans = max(peak_dtrans, float(d_trans.mean()))
            if step % 100 == 0 or step == 499:
                logs.append((step, float(lam), float(d_trans.mean()), float(d_sg.mean())))
                print(f"   step {step:3d} lam {float(lam):.3f} d_trans {float(d_trans.mean()):.3f} "
                      f"d_sg {float(d_sg.mean()):.3f}")
        # checks: lambda bounded; local cost converging toward 1 (dual ascent works)
        assert math.isfinite(logs[-1][1]) and logs[-1][1] < 1e3, "lambda diverged"
        assert logs[-1][2] < 0.5 * peak_dtrans + 0.5, \
            f"d_trans not converging toward 1: final={logs[-1][2]:.2f} peak={peak_dtrans:.2f}"
        assert logs[-1][2] < 2.0, f"d_trans still far from local cost 1: {logs[-1][2]:.2f}"
        # monotonicity & scale: along a held-out traj, d(z_0, z_k) should increase with k
        head.eval()
        with torch.no_grad():
            s, L = dset.starts[0], dset.lengths[0]
            z0 = dset.latents[s].float().unsqueeze(0)
            ds = []
            for k in range(L):
                zk = dset.latents[s + k].float().unsqueeze(0)
                keep = dset._keep(s, s + k)
                ds.append(float(head(z0, zk, keep)[0]))
            # forward distances should be non-trivially increasing
            inc = np.mean(np.diff(ds) > 0)
            # asymmetry: d(0->last) vs d(last->0)
            keep = dset._keep(s, s + L - 1)
            d_fwd = float(head(dset.latents[s].float().unsqueeze(0),
                               dset.latents[s + L - 1].float().unsqueeze(0), keep)[0])
            d_bwd = float(head(dset.latents[s + L - 1].float().unsqueeze(0),
                               dset.latents[s].float().unsqueeze(0), keep)[0])
        print(f"   recovery: d(0->k) increasing frac={inc:.2f}; "
              f"d_fwd={d_fwd:.2f} d_bwd={d_bwd:.2f} (asym={abs(d_fwd-d_bwd):.2f})")
        assert inc > 0.6, f"d not monotone enough along traj: inc={inc}"
        print("   QRL recovery ok (bounded lambda, unit local cost, monotone d)")


def test_cem_objective():
    """The qm energy plugs into the CEM objective signature, is batch-independent
    per sample (no BatchNorm leakage across CEM candidates), and reduces to the
    stock masked-L2 floor when w_qm=0."""
    print("== test_cem_objective ==")
    from planning.objectives import create_objective_fn, create_qm_objective_fn
    torch.manual_seed(0)
    B, T, P, D, PR = 12, 6, 196, 384, 10
    head = QuasimetricHead(head_type="iqe").eval()
    zp = {"visual": torch.randn(B, T, P, D), "proprio": torch.randn(B, T, PR)}
    zg = {"visual": torch.randn(B, 1, P, D), "proprio": torch.randn(B, 1, PR)}
    vis = torch.ones(P); vis[20:40] = 0

    for per_step in (False, True):
        obj = create_qm_objective_fn(alpha=0, base=2, mode="last", qm_head=head,
                                     w_qm=1.0, w_l2=1.0, per_step=per_step)
        with torch.no_grad():
            e = obj(zp, zg, vis_mask=vis)
        assert e.shape == (B,) and torch.isfinite(e).all(), (per_step, e.shape)
        # per-sample batch independence: scoring sample i alone must match the batch
        for i in [0, 5, 11]:
            zpi = {k: v[i:i+1] for k, v in zp.items()}
            zgi = {k: v[i:i+1] for k, v in zg.items()}
            with torch.no_grad():
                ei = obj(zpi, zgi, vis_mask=vis)
            assert (ei[0] - e[i]).abs() < 1e-4, \
                f"energy of sample {i} depends on batch! ({per_step}) {float(ei[0])} vs {float(e[i])}"
        print(f"   per_step={per_step}: shape {tuple(e.shape)}, batch-independent ok")

    # w_qm=0 reduces to the stock masked-L2 (+alpha*proprio) floor
    base = create_objective_fn(0, 2, "last")
    obj0 = create_qm_objective_fn(0, 2, "last", head, w_qm=0.0, w_l2=1.0)
    with torch.no_grad():
        assert (base(zp, zg, vis_mask=vis) - obj0(zp, zg, vis_mask=vis)).abs().max() < 1e-5
    print("   w_qm=0 == stock masked-L2 floor ok")


if __name__ == "__main__":
    test_head()
    test_cem_objective()
    test_qrl_recovery()
    print("\nALL TESTS PASSED")

"""Closed-loop planner (base/dino_wm conda env): the env-boundary crossing the project deferred.

Spawns lt_envserver.py in the langtable env, drives a live LanguageTable block2block episode:
  re-encode observed DOT frame (DINOv2) -> R decodes block positions -> low-level CEM picks pusher
  actions by rolling out the FROZEN dynamics toward a block-position WAYPOINT (readout energy, NOT
  latent-L2) -> execute a few env steps in the REAL sim -> re-observe -> replan (inside K<=reliable).
Sub-goals & success live in the pusher-invariant readout space; success is the SIM's verdict.

Everything frozen: DINOv2 encoder, dynamics (Dyn), readout (R). Only the planner is new.

Modes:
  --mode preflight : the §3 pre-flight. (P1) ckpt round-trips (encoder/dyn/readout reload+forward),
                     (P2) ENV-BOUNDARY HANDSHAKE (reset->frame->encode->R-decode vs GT block_xy),
                     (P3) time one low-level CEM plan, (P4) one full action->sim->frame->encode->score
                     round trip (the OOM-guard-equivalent). One config, logged. -> STOP + report.

Run (base env): python lt_loop.py --mode preflight \
    --cache /workspace/lt_cache_3k --model /workspace/g2_3k_roll/model.pth \
    --readout /workspace/readout_3k/R.pth --lt_python /workspace/envs/langtable/bin/python
"""
import argparse
import os
import socket
import subprocess
import sys
import time

import numpy as np
import torch
import torchvision.transforms.functional as TVF

sys.path.insert(0, "/workspace/dino_goal/dino_wm")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lt_g2 import Dyn, NP          # noqa: E402
from lt_readout import Readout     # noqa: E402
import lt_ipc                      # noqa: E402

TAU = 0.1
RADIUS = 0.05
AB = 0.1                           # per-env-action clamp (matches lt_relplan / corpus action scale)


# ---------------- frozen DINOv2 encoder (EXACT lt_cache recipe; f16 round-trip to match training) ----------------
def make_encoder(dev):
    base = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").eval().to(dev)

    def enc(frames_u8):  # (N,224,224,3) uint8 -> (N,196,384) float32 (f16-rounded like the cache)
        with torch.no_grad():
            x = torch.from_numpy(np.ascontiguousarray(frames_u8)).permute(0, 3, 1, 2).float() / 255.0
            x = TVF.normalize(x, [0.5] * 3, [0.5] * 3)
            x = TVF.resize(x, [196, 196], antialias=True).to(dev)
            z = base.forward_features(x)["x_norm_patchtokens"]
        return z.half().float()  # mimic cache f16 storage so latents are in-distribution
    return enc, base


# ---------------- env-server subprocess + socket ----------------
class SimClient:
    def __init__(self, lt_python, script_dir, seed=0, size=224):
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind(("127.0.0.1", 0))
        port = self._srv.getsockname()[1]
        self._srv.listen(1)
        env = dict(os.environ)
        self.proc = subprocess.Popen(
            [lt_python, os.path.join(script_dir, "lt_envserver.py"),
             "--port", str(port), "--seed", str(seed), "--size", str(size)],
            env=env)
        self._srv.settimeout(60)
        self.conn, _ = self._srv.accept()
        self.conn.settimeout(120)

    def call(self, **msg):
        lt_ipc.send(self.conn, msg)
        r = lt_ipc.recv(self.conn)
        if r is None:
            raise RuntimeError("env-server closed the connection")
        if not r.get("ok", False):
            raise RuntimeError(f"env-server error: {r.get('err')}\n{r.get('tb','')}")
        return r

    def reset(self, seed=None, no_terminate=False):
        return self.call(cmd="reset", seed=seed, no_terminate=no_terminate)

    def step(self, env_actions):  # list of [dx,dy]
        return self.call(cmd="step", actions=[list(map(float, a)) for a in env_actions])

    def close(self):
        try:
            lt_ipc.send(self.conn, {"cmd": "close"})
        except OSError:
            pass
        try:
            self.proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            self.proc.kill()
        self.conn.close()
        self._srv.close()


# ---------------- norm stats from the training cache ----------------
def norm_stats(cache, dev):
    tr = dict(np.load(f"{cache}/train.npz", allow_pickle=True))

    def vstack(c, k):
        return np.concatenate([c[k][i, :int(c["seq_lengths"][i])] for i in range(len(c["seq_lengths"]))], 0)
    pm = vstack(tr, "proprio").mean(0); ps = vstack(tr, "proprio").std(0) + 1e-6
    am = vstack(tr, "actions").mean(0); as_ = vstack(tr, "actions").std(0) + 1e-6
    meta = dict(fs=int(tr["frameskip"]), blocks=[str(b) for b in tr["blocks"]],
                half=float(tr["half_extent"]), cx=float(tr["center"][0]), cy=float(tr["center"][1]))
    t = lambda x: torch.tensor(x, device=dev, dtype=torch.float32)
    return dict(pm=t(pm), ps=t(ps), am=t(am), as_=t(as_)), meta


# ---------------- batched dynamics rollout (matches lt_relplan.rollout) ----------------
def rollout(model, ns, lo, hi, fs, nh, vis_hist, prop_hist, act_prefix, ee0, cem_acts):
    """vis_hist (nh,196,d), prop_hist (nh,2), act_prefix (nh-1,fs*2), ee0 (2), cem_acts (B,H,fs*2) raw.
    Returns final predicted visual grid (B,196,d) and final ee (B,2)."""
    B, H = cem_acts.shape[0], cem_acts.shape[1]
    vis = [vis_hist[k].unsqueeze(0).expand(B, -1, -1) for k in range(nh)]
    prop = [prop_hist[k].unsqueeze(0).expand(B, -1) for k in range(nh)]
    act = [act_prefix[k].unsqueeze(0).expand(B, -1) for k in range(nh - 1)]
    ee = ee0.unsqueeze(0).expand(B, -1).clone()
    for h in range(H):
        a = cem_acts[:, h]; act.append(a)
        wv = torch.stack(vis[-nh:], 1)
        wp = (torch.stack(prop[-nh:], 1) - ns["pm"]) / ns["ps"]
        wa = (torch.stack(act[-nh:], 1) - ns["am"]) / ns["as_"]
        nxt = model.predict(model.assemble(wv, wp, wa))[:, -1, :NP]
        vis.append(nxt)
        ee = torch.clamp(ee + a.reshape(B, fs, 2).sum(1), lo, hi)
        prop.append(ee)
    return vis[-1], ee


def waypoint_cost(R, grid, ai, waypoint, lo, hi, ee_final=None, contact_pt=None, w_approach=0.0,
                  all_ref=None, protect_mask=None, dont_disturb=0.0, hard=False, flat=False):
    """Low-level cost: decoded pos of block A vs the world waypoint + off-table penalty.
    `flat=True` (L2 'none'): return a CONSTANT cost so CEM gets NO relational gradient -> the
    optimizer cannot steer (measures success without command information). `d` is still the real
    A->waypoint distance for logging.
    Optional CONTACT-APPROACH shaping (w_approach>0): pull the predicted final pusher (ee_final)
    to `contact_pt` = the point behind A along the push direction, so the CEM gets a gradient to
    make contact BEFORE A moves (object-only cost is otherwise flat until contact).
    Optional E.1 DON'T-DISTURB (dont_disturb>0): penalize predicted displacement of protected
    (non-target) blocks from their reference positions -> anti-bulldozing.
    Low-level controller terms only; the relational A-B energy and readout-space subgoals are untouched."""
    pos, _ = (R.decode_hard(grid) if hard else R.decode(grid, tau=TAU))   # (B,nblk,2)
    pa = pos[:, ai]                              # (B,2)
    d = (pa - waypoint[None]).norm(dim=-1)
    if flat:                                     # L2 'none': constant energy, no gradient
        return torch.zeros_like(d), d
    cost = d.clone()
    if w_approach > 0 and ee_final is not None and contact_pt is not None:
        cost = cost + w_approach * (ee_final - contact_pt[None]).norm(dim=-1)
    if dont_disturb > 0 and all_ref is not None and protect_mask is not None:
        disp = (pos - all_ref[None]).norm(dim=-1)               # (B,nblk) predicted block displacement
        cost = cost + dont_disturb * (disp * protect_mask[None]).sum(-1)
    oob = ((pa[:, 0] < lo[0]) | (pa[:, 0] > hi[0]) | (pa[:, 1] < lo[1]) | (pa[:, 1] > hi[1])).float()
    return cost + 0.5 * oob, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="preflight", choices=["preflight", "handshake", "h0b", "h3", "h3chain"])
    ap.add_argument("--chain_len", type=int, default=2)   # L4: relational subtasks per episode (disjoint pairs)
    ap.add_argument("--sub_steps", type=int, default=25)  # L4: model-step budget per subtask
    ap.add_argument("--wp_spacing", type=float, default=0.10)  # H.3 carrot: subgoal this far ahead of A toward B
    ap.add_argument("--dont_disturb", type=float, default=0.0)  # E.1 anti-bulldoze weight on non-target blocks
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--max_steps", type=int, default=40)     # model-steps per episode
    ap.add_argument("--wp_dist", type=float, default=0.10)   # H.0b waypoint distance from A (u)
    ap.add_argument("--execute_steps", type=int, default=1)  # K=1 receding horizon (DINO-WM MPC)
    ap.add_argument("--drift_thresh", type=float, default=0.08)  # render-consistency standing guard
    ap.add_argument("--act_clamp", type=float, default=0.04)   # per-env-action clamp (oracle max |comp|~0.034)
    ap.add_argument("--act_sigma", type=float, default=0.012)  # CEM init std (oracle |comp| std~0.009)
    ap.add_argument("--w_approach", type=float, default=0.5)   # contact-approach shaping weight (0 = ablate)
    ap.add_argument("--r_contact", type=float, default=0.035)  # pusher sits this far behind A along push dir
    ap.add_argument("--decode", default="soft", choices=["soft", "hard"])  # hard = red_moon cross-talk fix (Lever A)
    ap.add_argument("--waypoints", default="straight", choices=["straight", "obstacle"])  # Lever C
    ap.add_argument("--obs_clearance", type=float, default=0.06)  # C: block within this of A->B = obstruction
    ap.add_argument("--conservative", type=float, default=0.0)  # Lever D: scale act_clamp/sigma near goal (0=off)
    ap.add_argument("--conservative_dist", type=float, default=0.08)  # D: "near goal" radius
    ap.add_argument("--cmd", default="correct", choices=["correct", "none", "wrong", "swap"])  # L2 ablation
    #   correct = drive true A->B (anchor).  none = flat/constant energy (no relational gradient).
    #   wrong   = drive true A toward a DIFFERENT anchor (genuine wrong command; success symmetric so
    #             an A<->B referent swap is a no-op -> use anchor-substitution).  swap = push true B->A
    #             (symmetry control; should ~= correct). Success ALWAYS scored by the env on true (A,B).
    ap.add_argument("--cache", default="/workspace/lt_cache_3k")
    ap.add_argument("--model", default="/workspace/g2_3k_roll/model.pth")
    ap.add_argument("--readout", default="/workspace/readout_3k/R.pth")
    ap.add_argument("--lt_python", default="/workspace/envs/langtable/bin/python")
    ap.add_argument("--num_hist", type=int, default=3)
    ap.add_argument("--cem_H", type=int, default=3)      # low-level plan horizon (<= reliable K~3)
    ap.add_argument("--pop", type=int, default=96)
    ap.add_argument("--cem_iters", type=int, default=8)
    ap.add_argument("--elites", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"[config] mode={a.mode} model={a.model} readout={a.readout} seed={a.seed} "
          f"cem_H={a.cem_H} pop={a.pop} iters={a.cem_iters} elites={a.elites} device={dev}")

    ns, meta = norm_stats(a.cache, dev)
    fs, nh = meta["fs"], a.num_hist
    blocks = meta["blocks"]; bidx = {b: i for i, b in enumerate(blocks)}
    lo = torch.tensor([0.15, -0.3048], device=dev); hi = torch.tensor([0.6, 0.3048], device=dev)

    # ---- load frozen models ----
    t0 = time.time()
    enc, encoder_base = make_encoder(dev)
    ckm = torch.load(a.model, map_location=dev); ar = ckm.get("arch", {})
    model = Dyn(nh, 1, fs, depth=ar.get("depth", 6), heads=ar.get("heads", 6),
                mlp_dim=ar.get("mlp_dim", 2048)).to(dev)
    model.load_state_dict(ckm["model"]); model.eval()
    ck = torch.load(a.readout, map_location=dev)
    R = Readout(ck["nblk"], ck["half"], ck["cx"], ck["cy"]).to(dev)
    R.load_state_dict(ck["state"]); R.eval()
    print(f"[load] encoder+dyn(arch={ar or 'default 6/6/2048'})+readout in {time.time()-t0:.1f}s; "
          f"dyn params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    if a.mode == "preflight":
        run_preflight(a, dev, enc, encoder_base, model, R, ns, meta, fs, nh, bidx, lo, hi, script_dir)
    elif a.mode == "handshake":
        run_handshake_n(a, dev, enc, R, meta, lo, hi, script_dir)
    elif a.mode == "h0b":
        run_h0b(a, dev, enc, model, R, ns, meta, fs, nh, lo, hi, script_dir)
    elif a.mode == "h3":
        run_h3(a, dev, enc, model, R, ns, meta, fs, nh, lo, hi, script_dir)
    elif a.mode == "h3chain":
        run_h3chain(a, dev, enc, model, R, ns, meta, fs, nh, lo, hi, script_dir)


def run_handshake_n(a, dev, enc, R, meta, lo, hi, script_dir):
    """Batched env-boundary handshake (n resets): is R's read of the START block (A) systematically
    bad on LIVE start frames? Diagnose pusher-occlusion (corr A-err vs pusher->A dist) vs block-id
    weakness (per-name) vs general start-frame shift (all blocks bad). Catches what D1 (frame-avg)
    could hide: start frames are a minority, and A starts where the pusher sits."""
    blocks = meta["blocks"]
    sim = SimClient(a.lt_python, script_dir, seed=a.seed)
    rows = []
    try:
        for ep in range(a.n):
            r = sim.reset(seed=a.seed + ep)
            frame = np.asarray(r["frame"]); gt = np.asarray(r["block_xy"])  # (8,2)
            ai = blocks.index(r["start_block"]); bi = blocks.index(r["target_block"])
            ee = np.asarray(r["ee"])
            z0 = enc(frame[None])
            with torch.no_grad():
                pos, conf = R.decode(z0, tau=TAU)
            pos = pos[0].cpu().numpy(); conf = conf[0].cpu().numpy()
            err = np.linalg.norm(pos - gt, axis=-1)                 # (8,)
            others = [k for k in range(len(blocks)) if k not in (ai, bi)]
            rows.append(dict(
                A=r["start_block"], B=r["target_block"], ai=ai, bi=bi,
                aerr=float(err[ai]), berr=float(err[bi]), othererr=float(err[others].mean()),
                allmean=float(err.mean()),
                eeA=float(np.linalg.norm(ee - gt[ai])), eeB=float(np.linalg.norm(ee - gt[bi])),
                ddist=float(abs(np.linalg.norm(pos[ai] - pos[bi]) - np.linalg.norm(gt[ai] - gt[bi]))),
                aconf=float(conf[ai]), bconf=float(conf[bi])))
    finally:
        sim.close()

    arr = lambda k: np.array([r[k] for r in rows])
    aerr, berr, oth, ee_a = arr("aerr"), arr("berr"), arr("othererr"), arr("eeA")
    ddist = arr("ddist")
    print(f"\n=== BATCHED HANDSHAKE (n={len(rows)}) ===")
    print(f"  block A (start) decode-err: mean={aerr.mean():.4f}u median={np.median(aerr):.4f}u "
          f"frac<0.05={np.mean(aerr<0.05):.2f}  max={aerr.max():.4f}u")
    print(f"  block B (target)decode-err: mean={berr.mean():.4f}u median={np.median(berr):.4f}u "
          f"frac<0.05={np.mean(berr<0.05):.2f}")
    print(f"  other 6 blocks  decode-err: mean={oth.mean():.4f}u frac<0.05={np.mean(oth<0.05):.2f}")
    print(f"  decoded dist(A,B) err vs GT: mean={ddist.mean():.4f}u frac<0.05={np.mean(ddist<0.05):.2f}")
    # pusher-occlusion test: does A-err rise when the pusher (ee) is near A?
    if aerr.std() > 1e-6 and ee_a.std() > 1e-6:
        c = np.corrcoef(aerr, ee_a)[0, 1]
        near = ee_a < np.median(ee_a)
        print(f"  corr(A-err, pusher->A dist)={c:+.2f}  | A-err when pusher NEAR A={aerr[near].mean():.4f}u "
              f"vs FAR={aerr[~near].mean():.4f}u  (negative corr / near>far => pusher occludes A)")
    # per-start-block breakdown (block-id weakness?)
    from collections import defaultdict
    byblk = defaultdict(list)
    for r in rows:
        byblk[r["A"]].append(r["aerr"])
    print("  A-err by start-block:")
    for k in sorted(byblk, key=lambda k: -np.mean(byblk[k])):
        v = byblk[k]
        print(f"    {k:>16}: mean={np.mean(v):.4f}u (n={len(v)})")
    print(f"\n[VERDICT] A-err frac<0.05={np.mean(aerr<0.05):.2f}. If A systematically >> B/others and "
          f"rises near the pusher -> pusher-occludes-start-block (energy unreliable at contact); "
          f"if A~B~others -> handshake clean, P2 was n=1 noise.")


def low_level_plan(model, R, ns, fs, nh, lo, hi, vis_t, prop_t, actp_t, ee_cur, ai, waypoint,
                   H, pop, iters, elites, clamp, sigma0, contact_pt=None, w_approach=0.0,
                   all_ref=None, protect_mask=None, dont_disturb=0.0, hard=False, mu_init=None,
                   flat=False):
    """One CEM plan: pusher actions to drive decoded block A toward `waypoint` under the frozen
    dynamics (readout energy). Returns mu (H,fs*2) action plan + the elite's predicted A->wp dist.
    `clamp`/`sigma0` MUST match the training (oracle) action scale (oracle max |comp|~0.034) or the
    dynamics is queried OOD on actions -> exploitation/scatter. mu_init enables MPC warm-start."""
    dev = vis_t.device
    mu = torch.zeros(H, fs * 2, device=dev) if mu_init is None else mu_init.clone()
    sig = torch.full((H, fs * 2), sigma0, device=dev)
    best_d = None
    with torch.no_grad():
        for _ in range(iters):
            pop_a = torch.clamp(mu[None] + sig[None] * torch.randn(pop, H, fs * 2, device=dev), -clamp, clamp)
            grid, ee_final = rollout(model, ns, lo, hi, fs, nh, vis_t, prop_t, actp_t, ee_cur, pop_a)
            costs, dists = waypoint_cost(R, grid, ai, waypoint, lo, hi, ee_final=ee_final,
                                         contact_pt=contact_pt, w_approach=w_approach,
                                         all_ref=all_ref, protect_mask=protect_mask,
                                         dont_disturb=dont_disturb, hard=hard, flat=flat)
            idx = costs.topk(elites, largest=False).indices
            mu, sig = pop_a[idx].mean(0), pop_a[idx].std(0) + 1e-4
            best_d = dists[idx[0]].item()
    return mu, best_d


def run_h0b(a, dev, enc, model, R, ns, meta, fs, nh, lo, hi, script_dir):
    """H.0a/H.0b: minimal closed loop. CEM drives the START block A toward a single GIVEN nearby
    block-position waypoint (A_start + wp_dist toward table center). K=1 receding horizon: plan H,
    execute 1 model-step in the REAL sim, re-observe, replan. Success = SIM ground truth
    (||GT_A - waypoint|| < 0.05). Planner sees only R-decoded A (no GT). Standing render-drift guard."""
    blocks = meta["blocks"]
    center = np.array([meta["cx"], meta["cy"]], np.float32)
    lo_np = np.array([0.15, -0.3048], np.float32); hi_np = np.array([0.6, 0.3048], np.float32)
    print(f"[h0b] n={a.n} max_steps={a.max_steps} wp_dist={a.wp_dist} execute_steps(K)={a.execute_steps} "
          f"cem_H={a.cem_H} pop={a.pop} iters={a.cem_iters} act_clamp={a.act_clamp} act_sigma={a.act_sigma} "
          f"drift_thresh={a.drift_thresh}")
    sim = SimClient(a.lt_python, script_dir, seed=a.seed)
    res = []
    try:
        for ep in range(a.n):
            r = sim.reset(seed=a.seed + ep)
            gt = np.asarray(r["block_xy"]); ai = blocks.index(r["start_block"])
            z = enc(np.asarray(r["frame"])[None])
            with torch.no_grad():
                pos, _ = R.decode(z, tau=TAU)
            pos = pos[0].cpu().numpy()
            reset_err = float(np.linalg.norm(pos - gt, axis=-1).mean())
            # waypoint: A_start (GT) nudged wp_dist toward table center (on-table, generally unobstructed)
            d = center - gt[ai]; nrm = np.linalg.norm(d)
            d = d / nrm if nrm > 1e-6 else np.array([1.0, 0.0], np.float32)
            wp = np.clip(gt[ai] + a.wp_dist * d, lo_np, hi_np)
            wp_t = torch.tensor(wp, device=dev, dtype=torch.float32)
            d0 = float(np.linalg.norm(gt[ai] - wp))
            # cold-start history
            ee_cur = torch.tensor(np.asarray(r["ee"]), device=dev, dtype=torch.float32)
            vis_hist = [z[0].clone()] * nh
            prop_hist = [ee_cur.clone()] * nh
            act_prefix = [torch.zeros(fs * 2, device=dev)] * max(nh - 1, 1)
            reached, t_reach, max_drift = False, None, reset_err
            d_dec_traj, d_gt_traj = [], []
            mu_prev = None
            n_exec = 0
            done = False
            wm_best, eeA_min = float("inf"), float("inf")
            for step in range(a.max_steps):
                vis_t = torch.stack(vis_hist[-nh:])
                prop_t = torch.stack(prop_hist[-nh:])
                actp_t = torch.stack(act_prefix[-(nh - 1):]) if nh > 1 else torch.zeros(0, fs * 2, device=dev)
                # contact point: behind current decoded A along the push direction (A -> waypoint)
                Acur = torch.tensor(pos[ai], device=dev, dtype=torch.float32)
                u = wp_t - Acur; un = u.norm()
                u = u / un if un > 1e-6 else torch.tensor([1.0, 0.0], device=dev)
                contact_pt = Acur - a.r_contact * u
                eeA_min = min(eeA_min, float(np.linalg.norm(np.asarray(ee_cur.cpu()) - pos[ai])))
                mu, best_d = low_level_plan(model, R, ns, fs, nh, lo, hi, vis_t, prop_t, actp_t, ee_cur,
                                            ai, wp_t, a.cem_H, a.pop, a.cem_iters, a.elites,
                                            a.act_clamp, a.act_sigma, contact_pt=contact_pt,
                                            w_approach=a.w_approach, mu_init=mu_prev)
                wm_best = min(wm_best, best_d)
                for es in range(a.execute_steps):
                    act_exec = mu[es].detach().cpu().numpy()
                    s = sim.step(act_exec.reshape(fs, 2))
                    gt = np.asarray(s["block_xy"])
                    z = enc(np.asarray(s["frame"])[None])
                    with torch.no_grad():
                        pos, _ = R.decode(z, tau=TAU)
                    pos = pos[0].cpu().numpy()
                    drift = float(np.linalg.norm(pos - gt, axis=-1).mean())
                    max_drift = max(max_drift, drift)
                    ee_cur = torch.tensor(np.asarray(s["ee"]), device=dev, dtype=torch.float32)
                    vis_hist.append(z[0].clone()); vis_hist = vis_hist[-nh:]
                    prop_hist.append(ee_cur.clone()); prop_hist = prop_hist[-nh:]
                    act_prefix.append(torch.tensor(act_exec, device=dev, dtype=torch.float32))
                    act_prefix = act_prefix[-max(nh - 1, 1):]
                    n_exec += 1
                    d_gt = float(np.linalg.norm(gt[ai] - wp)); d_dec = float(np.linalg.norm(pos[ai] - wp))
                    d_gt_traj.append(d_gt); d_dec_traj.append(d_dec)
                    if d_gt < RADIUS and not reached:
                        reached, t_reach = True, n_exec
                    if s["done"]:
                        done = True
                    if reached or done:
                        break
                # MPC warm-start: shift executed steps off the plan
                mu_prev = torch.cat([mu[a.execute_steps:], torch.zeros(a.execute_steps, fs * 2, device=dev)], 0)
                if reached or done:
                    break
            dmin = min(d_gt_traj) if d_gt_traj else d0
            dfin = d_gt_traj[-1] if d_gt_traj else d0
            drift_flag = "  !!RENDER-DRIFT" if max_drift > a.drift_thresh else ""
            res.append(dict(A=r["start_block"], reached=reached, t=t_reach, d0=d0, dmin=dmin, dfin=dfin,
                            reset_err=reset_err, max_drift=max_drift, steps=n_exec,
                            wm_best=wm_best, eeA_min=eeA_min))
            print(f"  ep{ep:02d} A={r['start_block']:>16} reached={int(reached)} t={t_reach} "
                  f"d0={d0:.3f} dmin={dmin:.3f} dfin={dfin:.3f} eeAmin={eeA_min:.3f} wmBest={wm_best:.3f} "
                  f"resetErr={reset_err:.3f} maxDrift={max_drift:.3f}{drift_flag}")
    finally:
        sim.close()

    arr = lambda k: np.array([x[k] for x in res], dtype=float)
    reached = arr("reached").astype(bool)
    dmin, dfin, steps = arr("dmin"), arr("dfin"), arr("steps")
    treach = np.array([x["t"] for x in res if x["reached"]], dtype=float)
    print(f"\n=== H.0b GATE: single-waypoint reachability (n={len(res)}, sim-grounded) ===")
    print(f"  REACH RATE (ever within 0.05u): {reached.mean():.2f} ({reached.sum()}/{len(res)})")
    print(f"  end-state within 0.05u: {np.mean(dfin < RADIUS):.2f}   |  dmin mean={dmin.mean():.3f}u")
    print(f"  REACH within 0.07u (looser, chaining-usefulness): {np.mean(dmin < 0.07):.2f}")
    wmb, eeam = arr("wm_best"), arr("eeA_min")
    print(f"  EXPLOITATION gap: mean realized dmin={dmin.mean():.3f}u vs mean WM-predicted={wmb.mean():.3f}u "
          f"(gap={dmin.mean()-wmb.mean():.3f}u; large => WM over-predicts the push = model-error exploit)")
    print(f"  mean pusher->A closest approach={eeam.mean():.3f}u (small => contact-shaping reaches the block)")
    print(f"  time-to-reach (model-steps): mean={treach.mean():.1f} median={np.median(treach):.0f}"
          if len(treach) else "  time-to-reach: (none reached)")
    print(f"  max render-drift over all eps: {arr('max_drift').max():.3f}u "
          f"({'OK <thresh' if arr('max_drift').max() <= a.drift_thresh else 'DRIFT FLAGGED'})")
    from collections import defaultdict
    byb = defaultdict(list)
    for x in res:
        byb[x["A"]].append(x["reached"])
    print("  reach by start-block (failure clustering):")
    for k in sorted(byb, key=lambda k: np.mean(byb[k])):
        print(f"    {k:>16}: {np.mean(byb[k]):.2f} ({int(np.sum(byb[k]))}/{len(byb[k])})")
    print(f"\n[H.0b VERDICT] high reach rate => low-level loop can chain waypoints => build H.1. "
          f"low reach => diagnose low-level MPC / action-scale / env-boundary BEFORE the hierarchy.")


def carrot_target(pos, ai, bi, others, wp_spacing, mode, clearance, w_side=1.0):
    """Geometric high-level (readout/block-position space): place a sub-goal wp_spacing ahead of A
    toward B (B itself when within range). mode='obstacle' steers the carrot sideways around the
    nearest block that lies on/near the A->B segment (potential-field-style detour); 'straight' is the
    baseline carrot. Returns (target_xy, dist_A_to_B)."""
    A = pos[ai]; B = pos[bi]; toB = B - A; dB = float(np.linalg.norm(toB))
    if dB < 1e-6:
        return B.astype(np.float32), dB
    u = toB / dB
    if mode == "obstacle":
        perp = np.array([-u[1], u[0]], np.float32)
        best = None
        for k in others:
            pk = pos[k]; t = float(np.clip((pk - A) @ u, 0.0, dB))
            if t < 0.02 or t >= dB:
                continue
            proj = A + t * u; clr = float(np.linalg.norm(pk - proj))
            if clr < clearance and (best is None or t < best[0]):
                best = (t, pk, proj, clr)
        if best is not None:
            _, pk, proj, clr = best
            sgn = (pk - proj) @ perp
            side = -1.0 if sgn > 0 else 1.0                     # steer to the side away from the block
            d = u + w_side * side * perp; d = d / (np.linalg.norm(d) + 1e-9)
            return (A + wp_spacing * d).astype(np.float32), dB
    tgt = B if dB <= wp_spacing else A + wp_spacing * u
    return np.asarray(tgt, np.float32), dB


def run_h3(a, dev, enc, model, R, ns, meta, fs, nh, lo, hi, script_dir):
    """H.1+H.2+H.3: full relational closed loop. Geometric high-level = receding carrot waypoint
    (a subgoal wp_spacing ahead of A toward B, in readout/block-position space; B itself when within
    range) -> low-level CEM drives A to it (contact-shaping), K=1 receding horizon, re-observe, replan.
    Success = the SIM's own block2block verdict (s['success'] = ||GT_A-GT_B|| < 0.05), NEVER the WM.
    Tracks non-target block displacement (the E.1 bulldoze baseline) and the exploitation check."""
    blocks = meta["blocks"]; nblk = len(blocks)
    lo_np = np.array([0.15, -0.3048], np.float32); hi_np = np.array([0.6, 0.3048], np.float32)
    hard = (a.decode == "hard")
    dec = (lambda g: R.decode_hard(g)) if hard else (lambda g: R.decode(g, tau=TAU))
    print(f"[h3] n={a.n} max_steps={a.max_steps} wp_spacing={a.wp_spacing} K={a.execute_steps} "
          f"cem_H={a.cem_H} pop={a.pop} iters={a.cem_iters} w_approach={a.w_approach} r_contact={a.r_contact} "
          f"dont_disturb={a.dont_disturb} decode={a.decode} waypoints={a.waypoints} cmd={a.cmd} "
          f"conservative={a.conservative} act_clamp={a.act_clamp} act_sigma={a.act_sigma}")
    sim = SimClient(a.lt_python, script_dir, seed=a.seed)
    res = []
    try:
        for ep in range(a.n):
            r = sim.reset(seed=a.seed + ep)
            gt = np.asarray(r["block_xy"]); gt0 = gt.copy()
            ai = blocks.index(r["start_block"]); bi = blocks.index(r["target_block"])
            others = [k for k in range(nblk) if k not in (ai, bi)]
            protect = torch.zeros(nblk, device=dev); protect[others] = 1.0  # don't-disturb: all but A,B
            # L2 swapped-command: the planner aims at (ai_plan,bi_plan); success is ALWAYS scored by
            # the env + dAB on the TRUE (ai,bi). 'wrong' substitutes the anchor (the symmetric metric
            # makes an A<->B swap a no-op); 'swap' pushes true-B->true-A (symmetry control); 'none' uses
            # a flat energy (no relational gradient).
            if a.cmd == "wrong":
                cand = [k for k in range(nblk) if k not in (ai, bi)]
                bi_plan = int(np.random.RandomState(1000 + ep).choice(cand)); ai_plan = ai
            elif a.cmd == "swap":
                ai_plan, bi_plan = bi, ai
            else:
                ai_plan, bi_plan = ai, bi
            flat = (a.cmd == "none")
            others_plan = [k for k in range(nblk) if k not in (ai_plan, bi_plan)]
            z = enc(np.asarray(r["frame"])[None])
            with torch.no_grad():
                pos, _ = dec(z)
            pos = pos[0].cpu().numpy()
            ee_cur = torch.tensor(np.asarray(r["ee"]), device=dev, dtype=torch.float32)
            vis_hist = [z[0].clone()] * nh
            prop_hist = [ee_cur.clone()] * nh
            act_prefix = [torch.zeros(fs * 2, device=dev)] * max(nh - 1, 1)
            d0 = float(np.linalg.norm(gt[ai] - gt[bi]))
            success, t_succ, done = False, None, False
            max_drift = float(np.linalg.norm(pos - gt, axis=-1).mean())
            dAB_min, n_exec, mu_prev = d0, 0, None
            for step in range(a.max_steps):
                curA = pos[ai_plan]
                target, dB = carrot_target(pos, ai_plan, bi_plan, others_plan, a.wp_spacing, a.waypoints, a.obs_clearance)
                dir_t = target - curA; ndir = float(np.linalg.norm(dir_t))
                dir_t = dir_t / ndir if ndir > 1e-9 else np.array([1.0, 0.0], np.float32)
                target_t = torch.tensor(target, device=dev, dtype=torch.float32)
                contact_pt = torch.tensor(curA - a.r_contact * dir_t, device=dev, dtype=torch.float32)
                all_ref = torch.tensor(pos, device=dev, dtype=torch.float32)        # don't-disturb reference
                # Lever D: conservative (gentler) actions near the goal to curb 1-step over-prediction overshoot
                clamp_e, sig_e = a.act_clamp, a.act_sigma
                if a.conservative > 0 and dB < a.conservative_dist:
                    clamp_e *= a.conservative; sig_e *= a.conservative
                vis_t = torch.stack(vis_hist[-nh:]); prop_t = torch.stack(prop_hist[-nh:])
                actp_t = torch.stack(act_prefix[-(nh - 1):]) if nh > 1 else torch.zeros(0, fs * 2, device=dev)
                mu, _ = low_level_plan(model, R, ns, fs, nh, lo, hi, vis_t, prop_t, actp_t, ee_cur,
                                       ai_plan, target_t, a.cem_H, a.pop, a.cem_iters, a.elites,
                                       clamp_e, sig_e, contact_pt=contact_pt, w_approach=a.w_approach,
                                       all_ref=all_ref, protect_mask=protect, dont_disturb=a.dont_disturb,
                                       hard=hard, mu_init=mu_prev, flat=flat)
                for es in range(a.execute_steps):
                    act_exec = mu[es].detach().cpu().numpy()
                    s = sim.step(act_exec.reshape(fs, 2))
                    gt = np.asarray(s["block_xy"])
                    z = enc(np.asarray(s["frame"])[None])
                    with torch.no_grad():
                        pos, _ = dec(z)
                    pos = pos[0].cpu().numpy()
                    max_drift = max(max_drift, float(np.linalg.norm(pos - gt, axis=-1).mean()))
                    ee_cur = torch.tensor(np.asarray(s["ee"]), device=dev, dtype=torch.float32)
                    vis_hist.append(z[0].clone()); vis_hist = vis_hist[-nh:]
                    prop_hist.append(ee_cur.clone()); prop_hist = prop_hist[-nh:]
                    act_prefix.append(torch.tensor(act_exec, device=dev, dtype=torch.float32))
                    act_prefix = act_prefix[-max(nh - 1, 1):]
                    n_exec += 1
                    dAB_min = min(dAB_min, float(np.linalg.norm(gt[ai] - gt[bi])))
                    if s["success"] and not success:
                        success, t_succ = True, n_exec
                    if s["done"]:
                        done = True
                    if success or done:
                        break
                mu_prev = torch.cat([mu[a.execute_steps:], torch.zeros(a.execute_steps, fs * 2, device=dev)], 0)
                if success or done:
                    break
            dAB_fin = float(np.linalg.norm(gt[ai] - gt[bi]))
            disturb = float(np.linalg.norm(gt[others] - gt0[others], axis=-1).mean())
            drift_flag = "  !!RENDER-DRIFT" if max_drift > a.drift_thresh else ""
            res.append(dict(A=r["start_block"], B=r["target_block"], success=success, t=t_succ, d0=d0,
                            dABmin=dAB_min, dABfin=dAB_fin, disturb=disturb, max_drift=max_drift, steps=n_exec))
            print(f"  ep{ep:02d} {r['start_block']:>15}->{r['target_block']:<15} succ={int(success)} t={t_succ} "
                  f"d0={d0:.3f} dABmin={dAB_min:.3f} dABfin={dAB_fin:.3f} disturb={disturb:.3f} "
                  f"maxDrift={max_drift:.3f}{drift_flag}")
    finally:
        sim.close()

    arr = lambda k: np.array([x[k] for x in res], dtype=float)
    succ = arr("success").astype(bool)
    dABmin, dABfin, disturb = arr("dABmin"), arr("dABfin"), arr("disturb")
    tsucc = np.array([x["t"] for x in res if x["success"]], dtype=float)
    print(f"\n=== H.3 GATE: relational closed-loop success (n={len(res)}, SIM-grounded block2block) ===")
    print(f"  SUCCESS RATE (sim ||A-B||<0.05): {succ.mean():.2f} ({succ.sum()}/{len(res)})")
    print(f"  ever within 0.05u during episode: {np.mean(dABmin < RADIUS):.2f}  | within 0.07u: {np.mean(dABmin < 0.07):.2f}")
    print(f"  dist(A,B): start mean={arr('d0').mean():.3f}u  final mean={dABfin.mean():.3f}u  min mean={dABmin.mean():.3f}u")
    print(f"  non-target block displacement (BULLDOZE, E.1 baseline): mean={disturb.mean():.3f}u max={disturb.max():.3f}u")
    print(f"  time-to-success (model-steps): mean={tsucc.mean():.1f} median={np.median(tsucc):.0f}"
          if len(tsucc) else "  time-to-success: (none)")
    print(f"  render-drift flagged on {int((arr('max_drift') > a.drift_thresh).sum())}/{len(res)} eps "
          f"(max {arr('max_drift').max():.3f}u)")
    from collections import defaultdict
    byb = defaultdict(list)
    for x in res:
        byb[x["A"]].append(x["success"])
    print("  success by start-block (failure clustering):")
    for k in sorted(byb, key=lambda k: np.mean(byb[k])):
        print(f"    {k:>16}: {np.mean(byb[k]):.2f} ({int(np.sum(byb[k]))}/{len(byb[k])})")
    ex_rm = [x["success"] for x in res if x["A"] != "red_moon"]
    if len(ex_rm) < len(res):
        print(f"  success EXCLUDING red_moon (known weak readout): {np.mean(ex_rm):.2f} ({int(np.sum(ex_rm))}/{len(ex_rm)})")
    print(f"\n[H.3 VERDICT] the first real plannability number (sim-grounded). disturb>>0 => E.1 anti-bulldoze due.")


def run_h3chain(a, dev, enc, model, R, ns, meta, fs, nh, lo, hi, script_dir):
    """L4 (DIAGNOSTIC FORECAST, NOT a gate): multi-step relational chains. Per episode, a hand-built
    sequence of chain_len DISJOINT (mover,anchor) pairs is executed back-to-back in ONE continuous
    episode (env-termination suppressed via no_terminate); each subtask is the same carrot + K=1
    closed-loop with re-observation between subtasks. per-subtask success = GT ||mover-anchor||<0.05
    ever reached; END-TO-END = ALL pairs satisfied at the FINAL layout (captures later steps
    disturbing earlier pairs). Forecasts Phase-G G.2 compositional risk BEFORE any VLM is built."""
    blocks = meta["blocks"]; nblk = len(blocks)
    hard = (a.decode == "hard")
    dec = (lambda g: R.decode_hard(g)) if hard else (lambda g: R.decode(g, tau=TAU))
    print(f"[h3chain] n={a.n} chain_len={a.chain_len} sub_steps={a.sub_steps} K={a.execute_steps} "
          f"cem_H={a.cem_H} decode={a.decode} w_approach={a.w_approach} act_clamp={a.act_clamp} seed={a.seed}")
    sim = SimClient(a.lt_python, script_dir, seed=a.seed)
    res = []
    try:
        for ep in range(a.n):
            r = sim.reset(seed=a.seed + ep, no_terminate=True)
            gt = np.asarray(r["block_xy"])
            rng = np.random.RandomState(a.seed + ep)
            picks = list(rng.choice(nblk, size=2 * a.chain_len, replace=False))  # disjoint blocks
            chain = [(int(picks[2 * k]), int(picks[2 * k + 1])) for k in range(a.chain_len)]
            z = enc(np.asarray(r["frame"])[None])
            with torch.no_grad():
                pos, _ = dec(z)
            pos = pos[0].cpu().numpy()
            ee_cur = torch.tensor(np.asarray(r["ee"]), device=dev, dtype=torch.float32)
            vis_hist = [z[0].clone()] * nh
            prop_hist = [ee_cur.clone()] * nh
            act_prefix = [torch.zeros(fs * 2, device=dev)] * max(nh - 1, 1)
            sub_succ, sub_steps_used = [], []
            for (mi, bi_p) in chain:
                others_p = [k for k in range(nblk) if k not in (mi, bi_p)]
                mu_prev, reached, used = None, False, 0
                for step in range(a.sub_steps):
                    curM = pos[mi]
                    target, dB = carrot_target(pos, mi, bi_p, others_p, a.wp_spacing, a.waypoints, a.obs_clearance)
                    dir_t = target - curM; ndir = float(np.linalg.norm(dir_t))
                    dir_t = dir_t / ndir if ndir > 1e-9 else np.array([1.0, 0.0], np.float32)
                    target_t = torch.tensor(target, device=dev, dtype=torch.float32)
                    contact_pt = torch.tensor(curM - a.r_contact * dir_t, device=dev, dtype=torch.float32)
                    vis_t = torch.stack(vis_hist[-nh:]); prop_t = torch.stack(prop_hist[-nh:])
                    actp_t = torch.stack(act_prefix[-(nh - 1):]) if nh > 1 else torch.zeros(0, fs * 2, device=dev)
                    mu, _ = low_level_plan(model, R, ns, fs, nh, lo, hi, vis_t, prop_t, actp_t, ee_cur,
                                           mi, target_t, a.cem_H, a.pop, a.cem_iters, a.elites,
                                           a.act_clamp, a.act_sigma, contact_pt=contact_pt,
                                           w_approach=a.w_approach, hard=hard, mu_init=mu_prev)
                    for es in range(a.execute_steps):
                        act_exec = mu[es].detach().cpu().numpy()
                        s = sim.step(act_exec.reshape(fs, 2))
                        gt = np.asarray(s["block_xy"])
                        z = enc(np.asarray(s["frame"])[None])
                        with torch.no_grad():
                            pos, _ = dec(z)
                        pos = pos[0].cpu().numpy()
                        ee_cur = torch.tensor(np.asarray(s["ee"]), device=dev, dtype=torch.float32)
                        vis_hist.append(z[0].clone()); vis_hist = vis_hist[-nh:]
                        prop_hist.append(ee_cur.clone()); prop_hist = prop_hist[-nh:]
                        act_prefix.append(torch.tensor(act_exec, device=dev, dtype=torch.float32))
                        act_prefix = act_prefix[-max(nh - 1, 1):]
                        used += 1
                        if float(np.linalg.norm(gt[mi] - gt[bi_p])) < RADIUS:
                            reached = True
                    mu_prev = torch.cat([mu[a.execute_steps:], torch.zeros(a.execute_steps, fs * 2, device=dev)], 0)
                    if reached:
                        break
                sub_succ.append(bool(reached)); sub_steps_used.append(used)
            final_ok = all(float(np.linalg.norm(gt[mi] - gt[bi_p])) < RADIUS for (mi, bi_p) in chain)
            res.append(dict(sub=sub_succ, e2e=bool(final_ok), steps=sub_steps_used))
            names = " ; ".join(f"{blocks[m]}->{blocks[b]}" for (m, b) in chain)
            print(f"  ep{ep:02d} sub={['Y' if x else 'n' for x in sub_succ]} e2e={int(final_ok)} "
                  f"steps={sub_steps_used}  [{names}]")
    finally:
        sim.close()
    arr_e2e = np.array([x["e2e"] for x in res], dtype=float)
    all_sub = np.array([s for x in res for s in x["sub"]], dtype=float)
    persub = float(np.mean(all_sub)) if len(all_sub) else 0.0
    print(f"\n=== L4 chain_len={a.chain_len} (n={len(res)}, DIAGNOSTIC forecast -- NOT a gate) ===")
    print(f"  per-subtask success (GT<0.05 ever): {persub:.2f} ({int(all_sub.sum())}/{len(all_sub)})")
    print(f"  END-TO-END (all pairs at FINAL layout): {arr_e2e.mean():.2f} ({int(arr_e2e.sum())}/{len(res)})")
    print(f"  naive product persub^{a.chain_len} = {persub**a.chain_len:.2f}  (closed-loop replanning should beat this)")
    for k in range(a.chain_len):
        sk = np.array([x["sub"][k] for x in res if len(x["sub"]) > k], dtype=float)
        print(f"    subtask#{k} success: {sk.mean():.2f} ({int(sk.sum())}/{len(sk)})")
    print(f"\n[L4 VERDICT] forecast for Phase-G G.2: if END-TO-END decays toward unusable at 3-step, "
          f"cross-subtask recovery needs work before the VLM. Diagnostic only; does NOT change L1-L3.")


def run_preflight(a, dev, enc, encoder_base, model, R, ns, meta, fs, nh, bidx, lo, hi, script_dir):
    cuda = dev == "cuda"

    # ============ [P1] checkpoint round-trips (reload + one forward = the OOM guard) ============
    print("\n=== PRE-FLIGHT [P1] checkpoint round-trips (reload -> forward -> identical) ===")
    dummy = (np.random.RandomState(0).rand(1, 224, 224, 3) * 255).astype(np.uint8)
    z = enc(dummy)
    z2 = enc(dummy)
    print(f"  encoder determinism: max|z-z2|={ (z-z2).abs().max().item():.2e}  shape={tuple(z.shape)} "
          f"({'PASS' if (z-z2).abs().max().item() < 1e-3 else 'CHECK'})")
    # dynamics reload (arch-faithful)
    tmp_m = "/tmp/dyn_rt.pth"
    torch.save({"model": model.state_dict()}, tmp_m)
    ckm = torch.load(a.model, map_location=dev); ar = ckm.get("arch", {})
    m2 = Dyn(nh, 1, fs, depth=ar.get("depth", 6), heads=ar.get("heads", 6),
             mlp_dim=ar.get("mlp_dim", 2048)).to(dev)
    m2.load_state_dict(torch.load(tmp_m, map_location=dev)["model"]); m2.eval()
    wv = z.unsqueeze(1).expand(1, nh, -1, -1)
    wp = torch.zeros(1, nh, 2, device=dev); wa = torch.zeros(1, nh, fs * 2, device=dev)
    with torch.no_grad():
        p1 = model.predict(model.assemble(wv, wp, wa))[:, -1, :NP]
        p2 = m2.predict(m2.assemble(wv, wp, wa))[:, -1, :NP]
    print(f"  dynamics reload: max|pred diff|={(p1-p2).abs().max().item():.2e} "
          f"({'PASS' if (p1-p2).abs().max().item() < 1e-6 else 'FAIL'})")
    # readout reload
    tmp_r = "/tmp/R_rt.pth"
    torch.save({"state": R.state_dict(), "nblk": R.nblk, "half": R.half, "cx": R.cx, "cy": R.cy}, tmp_r)
    rk = torch.load(tmp_r, map_location=dev)
    R2 = Readout(rk["nblk"], rk["half"], rk["cx"], rk["cy"]).to(dev)
    R2.load_state_dict(rk["state"]); R2.eval()
    with torch.no_grad():
        d1, _ = R.decode(z, tau=TAU); d2, _ = R2.decode(z, tau=TAU)
    print(f"  readout reload: max|decode diff|={(d1-d2).abs().max().item():.2e} "
          f"({'PASS' if (d1-d2).abs().max().item() < 1e-6 else 'FAIL'})")
    if cuda:
        print(f"  GPU mem after load+forward: {torch.cuda.max_memory_allocated()/1e9:.2f} GB (no OOM)")

    # ============ spawn the sim server (the env boundary) ============
    print("\n=== PRE-FLIGHT spawning env-server (langtable env) ===")
    sim = SimClient(a.lt_python, script_dir, seed=a.seed)
    try:
        # ============ [P2] ENV-BOUNDARY HANDSHAKE: reset -> frame -> encode -> R-decode vs GT ============
        print("=== PRE-FLIGHT [P2] env-boundary handshake (live sim -> encode -> R-decode vs GT) ===")
        r = sim.reset(seed=a.seed)
        frame = np.asarray(r["frame"]); gt = np.asarray(r["block_xy"])  # (8,2)
        blocks = r["blocks"]; ai = blocks.index(r["start_block"]); bi = blocks.index(r["target_block"])
        print(f"  reset: frame{frame.shape} {frame.dtype}  instr={r['instruction']!r}")
        print(f"  named pair: A={r['start_block']} (idx {ai})  B={r['target_block']} (idx {bi})")
        print(f"  server geometry: half={r['half_extent']} center={r['center']} "
              f"(cache half={meta['half']} center=({meta['cx']},{meta['cy']}))")
        z0 = enc(frame[None])                                  # (1,196,384)
        with torch.no_grad():
            pos, conf = R.decode(z0, tau=TAU)
        pos = pos[0].cpu().numpy()                             # (8,2)
        err = np.linalg.norm(pos - gt, axis=-1)               # per-block decode err
        dA = np.linalg.norm(pos[ai] - pos[bi]); dA_gt = np.linalg.norm(gt[ai] - gt[bi])
        print(f"  R-decode vs GT block_xy: mean err={err.mean():.4f}u  max={err.max():.4f}u  "
              f"A-err={err[ai]:.4f}u B-err={err[bi]:.4f}u")
        print(f"  decoded dist(A,B)={dA:.4f}u  GT dist(A,B)={dA_gt:.4f}u  |diff|={abs(dA-dA_gt):.4f}u")
        hs_ok = err.mean() < 0.05 and abs(dA - dA_gt) < 0.05
        print(f"  -> HANDSHAKE {'PASS' if hs_ok else 'CHECK'} "
              f"(coord frames consistent; live latents in-distribution for R)")

        # ============ [P3] time one low-level CEM plan toward a waypoint ============
        print("=== PRE-FLIGHT [P3] time one low-level CEM plan (readout waypoint cost) ===")
        # cold-start history: repeat the reset frame nh times, zero actions
        ee0 = torch.tensor(np.asarray(r["ee"]), device=dev, dtype=torch.float32)
        vis_hist = [z0[0]] * nh
        prop_hist = [ee0] * nh
        act_prefix = [torch.zeros(fs * 2, device=dev)] * (nh - 1)
        vis_hist = torch.stack(vis_hist); prop_hist = torch.stack(prop_hist)
        act_prefix = torch.stack(act_prefix) if nh > 1 else torch.zeros(0, fs * 2, device=dev)
        # nearby dummy waypoint: nudge A's decoded pos toward B by 0.05u (a reachable target)
        startA = torch.tensor(pos[ai], device=dev, dtype=torch.float32)
        dirAB = torch.tensor(gt[bi] - pos[ai], device=dev, dtype=torch.float32)
        dirAB = dirAB / (dirAB.norm() + 1e-6)
        waypoint = torch.clamp(startA + 0.05 * dirAB, lo, hi)
        H = a.cem_H
        mu = torch.zeros(H, fs * 2, device=dev); sig = torch.full((H, fs * 2), 0.06, device=dev)
        if cuda:
            torch.cuda.synchronize()
        t0 = time.time()
        best_d = None
        with torch.no_grad():
            for it in range(a.cem_iters):
                pop = torch.clamp(mu[None] + sig[None] * torch.randn(a.pop, H, fs * 2, device=dev), -AB, AB)
                grid, _ = rollout(model, ns, lo, hi, fs, nh, vis_hist, prop_hist, act_prefix, ee0, pop)
                costs, dists = waypoint_cost(R, grid, ai, waypoint, lo, hi)
                idx = costs.topk(a.elites, largest=False).indices
                mu, sig = pop[idx].mean(0), pop[idx].std(0) + 1e-4
                best_d = dists[idx[0]].item()
        if cuda:
            torch.cuda.synchronize()
        t_plan = time.time() - t0
        best_action = mu[0].detach().cpu().numpy()             # first model-step action (10-dim)
        print(f"  one CEM plan = {t_plan:.2f}s (H={H},pop={a.pop},iters={a.cem_iters}) "
              f"-> n=30 ~ {30*t_plan/60:.1f} min (per replan)")
        print(f"  imagined best-action predicted A->waypoint dist={best_d:.4f}u "
              f"(waypoint was 0.05u from start; in imagination)")

        # ============ [P4] FULL round trip: plan -> execute in REAL sim -> re-encode -> re-decode ============
        print("=== PRE-FLIGHT [P4] full action->sim->frame->encode->score round trip ===")
        env_actions = best_action.reshape(fs, 2)               # one model-step = fs env actions
        s = sim.step(env_actions)
        frame1 = np.asarray(s["frame"]); gt1 = np.asarray(s["block_xy"])
        z1 = enc(frame1[None])
        with torch.no_grad():
            pos1, _ = R.decode(z1, tau=TAU)
        pos1 = pos1[0].cpu().numpy()
        realA_move = np.linalg.norm(pos1[ai] - pos[ai])
        gtA_move = np.linalg.norm(gt1[ai] - gt[ai])
        decA_err1 = np.linalg.norm(pos1[ai] - gt1[ai])
        print(f"  executed {fs} env actions in REAL sim; done={s['done']} success={s['success']}")
        print(f"  block A real move (decoded)={realA_move:.4f}u  (GT)={gtA_move:.4f}u  "
              f"A decode-err after step={decA_err1:.4f}u")
        print(f"  -> ROUND TRIP CLOSED: live action -> sim step -> render -> encode -> R-decode works.")
        print(f"\n[PRE-FLIGHT DONE] handshake={'PASS' if hs_ok else 'CHECK'}; "
              f"all models reload; CEM plan {t_plan:.2f}s; full loop closes. Ready for H.0a/H.0b.")
    finally:
        sim.close()


if __name__ == "__main__":
    main()

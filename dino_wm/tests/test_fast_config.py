"""Fast-config bundle unit tests (docs/PLANNING_SPEED_PROFILE.md "Fast config bundle").

CPU-only; no checkpoints, env, or GPU needed. Run from the dino_wm/ directory:
    /Users/Tom/miniforge3/envs/dino_wm_dev/bin/python tests/test_fast_config.py
(direct run -- the dev env has no pytest; the __main__ block runs every test_*)

Covers:
1. SDPA attention == naive attention in eval mode (the only mode it activates in).
2. SDPA flag is a NO-OP while dropout is active (train mode) -- RNG stream untouched.
3. Old-checkpoint compatibility: Attention instances WITHOUT the use_sdpa attribute
   (simulating unpickled pre-change checkpoints) run the stock path and accept
   enable_sdpa().
4. CEM fast branch (traj_chunk>1) reproduces the stock sequential loop exactly on CPU
   (elementwise mock dynamics -> bitwise-equal mu/sigma), with patch masks in play.
5. CEM skip_succeeded: skipped trajs keep frozen mu; remaining trajs get EXACTLY the
   actions they would have gotten anyway (candidate RNG stream preserved).
6. Config sanity: the new keys exist with stock-behavior defaults.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parents[1]))

from models.vit import Attention, ViTPredictor, enable_sdpa  # noqa: E402
from planning.cem import CEMPlanner  # noqa: E402
from planning.objectives import create_objective_fn  # noqa: E402

REPO = Path(__file__).parents[1]


# ---------------------------------------------------------------- mocks for CEM tests
class _DummyWandb:
    def log(self, *a, **k):
        pass


class _IdentityPreprocessor:
    def transform_obs(self, obs):
        return {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in obs.items()}


class _TinyWM(torch.nn.Module):
    """Deterministic, per-sample-independent mock dynamics.

    Elementwise update only -> batched and per-traj rollouts are BITWISE identical on
    CPU, which is what lets the chunk-equivalence test assert exact equality.
    """

    def __init__(self, P=3, D=4, prop=2):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.P, self.D, self.prop = P, D, prop

    def encode_obs(self, obs):
        return {"visual": obs["visual"] * 1.0, "proprio": obs["proprio"] * 1.0}

    def rollout_from_zobs(self, z_obs_0, act):
        v = z_obs_0["visual"][:, 0]    # (B, P, D)
        p = z_obs_0["proprio"][:, 0]   # (B, prop)
        vs, ps = [v], [p]
        for t in range(act.shape[1]):
            a = act[:, t]
            delta = a.sum(-1, keepdim=True).unsqueeze(-1)        # (B,1,1)
            vs.append(vs[-1] * 0.9 + 0.1 * delta)
            ps.append(ps[-1] * 0.9 + 0.1 * a[:, : self.prop])
        z = {"visual": torch.stack(vs, 1), "proprio": torch.stack(ps, 1)}
        return z, None


def _make_planner(n_evals, P=3, D=4, prop=2, **kw):
    wm = _TinyWM(P=P, D=D, prop=prop)
    planner = CEMPlanner(
        horizon=3,
        topk=3,
        num_samples=8,
        var_scale=1.0,
        opt_steps=2,
        eval_every=999,
        wm=wm,
        action_dim=2,
        objective_fn=create_objective_fn(alpha=1, base=2, mode="last"),
        preprocessor=_IdentityPreprocessor(),
        evaluator=None,
        wandb_run=_DummyWandb(),
        log_filename=None,
        **kw,
    )
    # exercise the manipulator-mask path too: drop one patch for every traj
    mask = torch.ones(n_evals, P)
    mask[:, 0] = 0
    planner.patch_mask = mask
    return planner


def _obs(n_evals, P=3, D=4, prop=2, seed=7):
    g = torch.Generator().manual_seed(seed)
    obs_0 = {
        "visual": torch.randn(n_evals, 1, P, D, generator=g),
        "proprio": torch.randn(n_evals, 1, prop, generator=g),
    }
    obs_g = {
        "visual": torch.randn(n_evals, 1, P, D, generator=g),
        "proprio": torch.randn(n_evals, 1, prop, generator=g),
    }
    return obs_0, obs_g


# -------------------------------------------------------------------- vit.py: SDPA
def _tiny_predictor(dropout=0.1):
    torch.manual_seed(0)
    return ViTPredictor(
        num_patches=4, num_frames=3, dim=16, depth=2, heads=2, mlp_dim=32,
        dropout=dropout, emb_dropout=0.0,
    )


def test_sdpa_matches_naive_in_eval():
    model = _tiny_predictor(dropout=0.1).eval()
    for T in (4, 8, 12):  # 1, 2, 3 frames of 4 patches
        x = torch.randn(5, T, 16)
        with torch.no_grad():
            enable_sdpa(model, False)
            out_naive = model(x)
            n = enable_sdpa(model, True)
            out_sdpa = model(x)
        assert n == 2, f"expected 2 Attention blocks, got {n}"
        assert torch.allclose(out_naive, out_sdpa, atol=1e-5, rtol=1e-4), (
            f"SDPA != naive at T={T}: max diff "
            f"{(out_naive - out_sdpa).abs().max().item():.3e}"
        )


def test_sdpa_flag_is_noop_under_active_dropout():
    model = _tiny_predictor(dropout=0.1).train()
    x = torch.randn(5, 12, 16)
    enable_sdpa(model, True)
    torch.manual_seed(123)
    out_flag_on = model(x)
    enable_sdpa(model, False)
    torch.manual_seed(123)
    out_flag_off = model(x)
    assert torch.equal(out_flag_on, out_flag_off), (
        "use_sdpa changed train-mode output: the dropout guard failed (RNG hazard)"
    )


def test_sdpa_old_checkpoint_instances_without_attr():
    model = _tiny_predictor(dropout=0.1).eval()
    for m in model.modules():
        if isinstance(m, Attention):
            del m.use_sdpa  # simulate an instance unpickled from a pre-change ckpt
    x = torch.randn(2, 12, 16)
    with torch.no_grad():
        out_old = model(x)          # getattr default False -> stock path, no crash
        n = enable_sdpa(model, True)
        out_sdpa = model(x)
    assert n == 2
    assert torch.allclose(out_old, out_sdpa, atol=1e-5, rtol=1e-4)


# -------------------------------------------------------------------- cem.py: fast branch
def test_traj_chunk_matches_sequential():
    n_evals = 4
    obs_0, obs_g = _obs(n_evals)

    torch.manual_seed(42)
    mu_seq, _ = _make_planner(n_evals).plan(obs_0, obs_g)

    for chunk in (2, 3, 64):  # uneven and oversize chunks included
        torch.manual_seed(42)
        mu_chunk, _ = _make_planner(n_evals, traj_chunk=chunk).plan(obs_0, obs_g)
        assert torch.equal(mu_seq, mu_chunk), (
            f"traj_chunk={chunk} diverged from sequential: max diff "
            f"{(mu_seq - mu_chunk).abs().max().item():.3e}"
        )


def test_skip_succeeded_freezes_skipped_and_preserves_others():
    n_evals = 4
    obs_0, obs_g = _obs(n_evals)

    torch.manual_seed(42)
    mu_base, _ = _make_planner(n_evals).plan(obs_0, obs_g)

    planner = _make_planner(n_evals, skip_succeeded=True)
    planner.success_mask = np.array([True, False, False, False])
    torch.manual_seed(42)
    mu_skip, _ = planner.plan(obs_0, obs_g)

    # skipped traj: mu frozen at init (zeros -- init_mu_sigma pads with zeros)
    assert torch.equal(mu_skip[0], torch.zeros_like(mu_skip[0])), "skipped traj was planned"
    # active trajs: byte-identical to the no-skip run (candidate RNG stream preserved)
    assert torch.equal(mu_base[1:], mu_skip[1:]), (
        "skip_succeeded perturbed the remaining trajectories' results"
    )


def test_skip_and_chunk_together():
    """The canonical FAST config enables BOTH knobs; pin their interaction (the
    active-list filtering composed with chunk slicing)."""
    n_evals = 5
    obs_0, obs_g = _obs(n_evals)

    torch.manual_seed(42)
    mu_base, _ = _make_planner(n_evals).plan(obs_0, obs_g)

    planner = _make_planner(n_evals, skip_succeeded=True, traj_chunk=2)
    planner.success_mask = np.array([False, True, False, True, False])
    torch.manual_seed(42)
    mu_both, _ = planner.plan(obs_0, obs_g)

    for t in (1, 3):  # skipped: frozen at init
        assert torch.equal(mu_both[t], torch.zeros_like(mu_both[t])), f"traj {t} planned"
    for t in (0, 2, 4):  # active: identical to the no-skip sequential run
        assert torch.equal(mu_base[t], mu_both[t]), f"traj {t} diverged"


def test_fast_branch_requires_fast_encode():
    try:
        _make_planner(2, traj_chunk=2, fast_encode=False)
    except ValueError:
        return
    raise AssertionError("traj_chunk>1 with fast_encode=false should raise")


# -------------------------------------------------------------------- config sanity
def test_config_defaults_are_stock():
    from omegaconf import OmegaConf

    mpc_cem = OmegaConf.load(REPO / "conf/planner/mpc_cem.yaml")
    assert mpc_cem.sub_planner.skip_succeeded is False
    assert mpc_cem.sub_planner.traj_chunk == 1

    for name in ("conf/plan_pusht.yaml", "conf/plan_pusht_multicolor.yaml"):
        cfg = OmegaConf.load(REPO / name)
        assert cfg.fast_tf32 is False, name
        assert cfg.fast_sdpa is False, name
        assert cfg.plan_eval_mode is False, name
        assert cfg.planner.sub_planner.skip_succeeded is False, name
        assert cfg.planner.sub_planner.traj_chunk == 1, name
        assert cfg.planner.sub_planner.eval_every == 1, name
        assert cfg.planner.max_iter == 10, name

    # every shipped MPCPlanner config must carry a FINITE max_iter (plan.py now fails
    # fast on inf; null in stock configs was the 12h-non-termination footgun)
    for name in ("conf/plan_pusht.yaml", "conf/plan_pusht_multicolor.yaml",
                 "conf/plan_point_maze.yaml", "conf/plan_wall.yaml"):
        cfg = OmegaConf.load(REPO / name)
        if cfg.planner.get("_target_") == "planning.mpc.MPCPlanner":
            assert cfg.planner.max_iter is not None and np.isfinite(cfg.planner.max_iter), name
    for name in ("conf/planner/mpc_cem.yaml", "conf/planner/mpc_gd.yaml"):
        cfg = OmegaConf.load(REPO / name)
        assert cfg.max_iter is not None and np.isfinite(cfg.max_iter), name


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")

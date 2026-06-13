"""Phase-3 Stage-2: CEM planning toward the COORDINATE bridge `g`'s synthesized goal.

Clean-scene pivot (Option A). The frozen stock PushT dynamics + CEM plan on the base
single-T scene (constant inert green goal-T, no decals) toward a goal latent, under the
DEPLOYABLE masked energy (objective.alpha=0 + mask_pusher=true: drop the goal-frame pusher
patches). Success = the env block-pose gate vs the held-out spec (pos<20px AND |ang|<20deg,
pose_only_success). Evaluates on the held-out TEST specs (novel decorrelated pose pairs) so
this is true generalization.

goal_source:
  oracle       : plan toward enc(real teleport frame block@spec) -- the ~0.80 deployable ceiling.
  bridge       : plan toward g.forward_coord(z_start, spec) -- the headline.
  swapped_spec : g with the spec rolled across the batch -- wrong-coordinate floor (success vs the
                 TRUE spec should collapse; an interpretable failure = block goes to the wrong place).
  random       : each eval gets another eval's real goal latent -- unrelated-goal lower bound.

Reuses plan.py's PlanWorkspace (planner/evaluator/mask/success untouched); only target prep
and the goal-latent override are coord-specific (the z_obs_g_override seam is front-end-agnostic).

Box (4090):
  DATASET_DIR=/workspace/data /workspace/envs/dino_wm/bin/python plan_coord.py \
    model_name=pusht ckpt_base_path=/workspace/ckpts n_evals=30 \
    goal_source=bridge coord.bridge_ckpt=$(pwd)/outputs/bridge/g_coord/g_best.pth \
    mask_pusher=true objective.alpha=0 coord.mask_tag=_masked \
    plan_eval_mode=true fast_tf32=true fast_sdpa=true \
    planner.sub_planner.eval_every=999 planner.sub_planner.skip_succeeded=true planner.sub_planner.traj_chunk=4
"""
import os
import warnings
from pathlib import Path

import gym
import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf, open_dict
from hydra.utils import to_absolute_path

import custom_resolvers  # noqa: F401
from env.venv import SubprocVectorEnv
from utils import seed, move_to_device
from planning.mpc import MPCPlanner
from plan import PlanWorkspace, load_model, apply_fast_flags

warnings.filterwarnings("ignore")


class CoordPlanWorkspace(PlanWorkspace):
    def __init__(self, cfg_dict, wm, dset, env, frameskip, wandb_run):
        self.coord = cfg_dict["coord"]
        super().__init__(cfg_dict=cfg_dict, wm=wm, dset=dset, env=env,
                         env_name="pusht", frameskip=frameskip, wandb_run=wandb_run)
        if cfg_dict["goal_source"] != "oracle":
            self._attach_goal_override()   # after the planner exists

    # --- held-out (init_state, spec) eval set ---------------------------------
    def _load_held_out_specs(self):
        d = Path(to_absolute_path(self.coord["data_path"])) / self.coord["split"]
        init = torch.load(d / "init_states.pth").numpy().astype(np.float64)   # (N,7)
        spec = torch.load(d / "specs.pth").numpy().astype(np.float64)         # (N,3)
        dist = np.linalg.norm(spec[:, :2] - init[:, 2:4], axis=1)
        mgd = self.coord.get("max_goal_dist")
        if mgd is not None:
            keep = dist <= float(mgd)
            init, spec = init[keep], spec[keep]
            print(f"[coord] reachability filter A->B dist<= {mgd}px: kept {keep.sum()}/{len(keep)}")
        assert len(init) >= self.n_evals, f"only {len(init)} specs available, need n_evals={self.n_evals}"
        init, spec = init[:self.n_evals], spec[:self.n_evals]
        goal = init.copy()
        goal[:, 2:5] = spec                                   # teleport block -> spec, pusher@start
        d2 = np.linalg.norm(spec[:, :2] - init[:, 2:4], axis=1)
        print(f"[coord] {self.n_evals} held-out '{self.coord['split']}' specs | "
              f"A->B dist mean {d2.mean():.0f}px (min {d2.min():.0f} max {d2.max():.0f})")
        return init, goal

    def prepare_targets(self):
        init_states, goal_states = self._load_held_out_specs()
        obs_0, state_0 = self.env.prepare(self.eval_seed, init_states)
        obs_g, state_g = self.env.prepare(self.eval_seed, goal_states)
        self.obs_0 = {k: np.expand_dims(v, axis=1) for k, v in obs_0.items()}
        self.obs_g = {k: np.expand_dims(v, axis=1) for k, v in obs_g.items()}
        self.state_0 = state_0
        self.state_g = state_g
        self.gt_actions = None
        # deployable mask: drop the goal-frame pusher patches (teleport keeps pusher@start,
        # so goal and real proxy coincide). Populated for the base PlanWorkspace mask path.
        gp = np.asarray(state_g, dtype=np.float64)[:, 0:2].copy()
        self.goal_pusher_xy = gp
        self.real_pusher_xy = gp.copy()
        self.coord_specs = np.asarray(state_g, dtype=np.float64)[:, 2:5].copy()   # (N,3) block@spec

    # --- goal-latent override (bridge / swapped_spec / random) ----------------
    def _load_coord_g(self):
        from models.bridge import BridgeG
        ckpt = self.coord.get("bridge_ckpt")
        if not ckpt:
            raise ValueError("goal_source=bridge/swapped_spec requires coord.bridge_ckpt")
        ck = torch.load(to_absolute_path(ckpt), map_location=self.device)
        c = ck["config"]
        if c.get("cond_mode") != "coord":
            raise ValueError(f"coord.bridge_ckpt has cond_mode={c.get('cond_mode')!r} != 'coord'")
        g = BridgeG(dim=c["dim"], depth=c["depth"], heads=c["heads"], cond_mode="coord",
                    n_freq=c.get("n_freq", 12), heat_sigma=c.get("heat_sigma", 1.2)).to(self.device)
        g.load_state_dict(ck["state_dict"])
        g.eval()
        print(f"[coord] loaded g (epoch {ck.get('epoch')}, changed-cos {ck.get('val_changed_cos')}) from {ckpt}")
        return g

    def _attach_goal_override(self):
        gs = self.cfg_dict["goal_source"]
        target = self.planner.sub_planner if isinstance(self.planner, MPCPlanner) else self.planner
        if not hasattr(target, "z_obs_g_override"):
            raise TypeError(f"goal_source={gs} requires a planner with z_obs_g_override; "
                            f"got {type(target).__name__}")
        trans_obs_0 = move_to_device(self.data_preprocessor.transform_obs(self.obs_0), self.device)
        trans_obs_g = move_to_device(self.data_preprocessor.transform_obs(self.obs_g), self.device)
        with torch.no_grad():
            z0 = self.wm.encode_obs(trans_obs_0)
            zg = self.wm.encode_obs(trans_obs_g)        # proprio placeholder (+ real-teleport ref)

        if gs == "random":
            zg["visual"] = torch.roll(zg["visual"], shifts=1, dims=0)   # another eval's real goal
            target.z_obs_g_override = zg
            print("[coord] random floor: rolled real-teleport goal latents (unrelated to each spec)")
            return

        g = self._load_coord_g()
        spec = torch.tensor(self.coord_specs, dtype=torch.float32, device=self.device)  # (N,3)
        if gs == "swapped_spec":
            spec = torch.roll(spec, shifts=1, dims=0)
            print("[coord] swapped_spec floor: rolled the coordinate spec (success measured vs TRUE spec)")
        with torch.no_grad():
            z_goal_vis = g.forward_coord(z0["visual"][:, 0], spec)      # (N,196,384)
        ref_cos = F.cosine_similarity(
            z_goal_vis.reshape(self.n_evals, -1),
            zg["visual"][:, 0].reshape(self.n_evals, -1), dim=-1)
        zg["visual"] = z_goal_vis.unsqueeze(1).to(zg["visual"].dtype)
        target.z_obs_g_override = zg
        print(f"[coord] {gs}: z_obs_g_override <- g | cos(g_goal, real-teleport) "
              f"mean={ref_cos.mean():.3f} min={ref_cos.min():.3f}")
        alpha = (self.cfg_dict.get("objective") or {}).get("alpha")
        if alpha not in (0, 0.0):
            print(f"[coord] WARN: objective.alpha={alpha} != 0 scores proprio vs a placeholder "
                  f"pusher; the deployable energy is alpha=0 + mask_pusher=true")


def planning_main_coord(cfg_dict):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    coord = cfg_dict["coord"]
    if cfg_dict["goal_source"] in ("bridge", "swapped_spec"):
        _p = coord.get("bridge_ckpt")
        if not _p or not os.path.exists(to_absolute_path(_p)):
            raise FileNotFoundError(
                f"goal_source={cfg_dict['goal_source']} requires an existing coord.bridge_ckpt (got {_p!r})")

    model_path = f"{cfg_dict['ckpt_base_path']}/outputs/{cfg_dict['model_name']}/"
    model_cfg = OmegaConf.load(os.path.join(model_path, "hydra.yaml"))
    seed(cfg_dict["seed"])

    _, dset = hydra.utils.call(model_cfg.env.dataset, num_hist=model_cfg.num_hist,
                               num_pred=model_cfg.num_pred, frameskip=model_cfg.frameskip)
    dset = dset["valid"]

    model_ckpt = Path(model_path) / "checkpoints" / f"model_{cfg_dict['model_epoch']}.pth"
    model = load_model(model_ckpt, model_cfg, model_cfg.num_action_repeat, device=device)
    if cfg_dict.get("plan_eval_mode", False):
        model.eval()
        print("[plan] plan_eval_mode=True -> model.eval()")
    apply_fast_flags(cfg_dict, model)

    pose_only = bool(cfg_dict.get("pose_only_success", True))
    env = SubprocVectorEnv([
        (lambda: gym.make("pusht", with_velocity=True, with_target=True,
                          pose_only_success=pose_only))
        for _ in range(cfg_dict["n_evals"])
    ])

    ws = CoordPlanWorkspace(cfg_dict, model, dset, env, model_cfg.frameskip, wandb_run=None)
    logs = ws.perform_planning()
    headline = {k: v for k, v in logs.items() if ("success" in k or "dist" in k)}
    print(f"\n=== COORD {cfg_dict['goal_source'].upper()} RESULT (n={cfg_dict['n_evals']}, "
          f"masked={cfg_dict.get('mask_pusher')}, alpha={(cfg_dict.get('objective') or {}).get('alpha')}) ===")
    print(headline)
    return logs


@hydra.main(config_path="conf", config_name="plan_pusht_coord", version_base=None)
def main(cfg: OmegaConf):
    with open_dict(cfg):
        cfg["saved_folder"] = os.getcwd()
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_dict["wandb_logging"] = False
    planning_main_coord(cfg_dict)


if __name__ == "__main__":
    main()

"""Phase-3 Stage-1/2: CEM planning toward rigid-transform LANGUAGE goals (clean scene).

Plans on the RETRAINED clean dynamics (outputs/clean_retrain) toward a goal latent under
the deployable masked energy (objective.alpha=0 + mask_pusher: drop the goal-frame pusher
patches), scored by the REGIONAL metric (block CENTROID in the same 3x3 cell + |dangle|<
pi/9, via env success_mode=regional). The scene is CLEAN (green-T removed, with_target=
False) -- matching what the retrained dynamics and the eventual g pipeline see. Evaluates
the held-out TEST split (disjoint trajectory pool) stratified across the 9 regions.

goal_source:
  oracle       : plan toward enc(real clean goal frame) -- the deployable ceiling (3.1).
  random       : another eval's goal latent -- unrelated-goal floor.
  bridge / swapped_spec : wired in Phase 4 (need the language-g).

Box (4090):
  DATASET_DIR=/workspace/data python plan_rigid.py model_name=clean_retrain \
    ckpt_base_path=/workspace/dino_goal/dino_wm n_evals=36 goal_source=oracle \
    mask_pusher=true objective.alpha=0 plan_eval_mode=true fast_tf32=true fast_sdpa=true \
    planner.sub_planner.traj_chunk=2 planner.sub_planner.num_samples=300 \
    planner.sub_planner.opt_steps=30 planner.max_iter=10
"""
import os
import warnings
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path

import gym
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from hydra.utils import to_absolute_path

import custom_resolvers  # noqa: F401
from env.venv import SubprocVectorEnv
from utils import seed, move_to_device
from planning.mpc import MPCPlanner
from plan import PlanWorkspace, load_model, apply_fast_flags
from metrics.regional_success import region_name

warnings.filterwarnings("ignore")


class RigidPlanWorkspace(PlanWorkspace):
    def __init__(self, cfg_dict, wm, dset, env, frameskip, wandb_run):
        self.rigid = cfg_dict["rigid"]
        self.region_cells = None
        super().__init__(cfg_dict=cfg_dict, wm=wm, dset=dset, env=env,
                         env_name="pusht", frameskip=frameskip, wandb_run=wandb_run)
        if cfg_dict["goal_source"] != "oracle":
            self._attach_goal_override()

    def _stratified_order(self, cells, n):
        """Round-robin across the 9 region cells so every region is represented."""
        buckets = OrderedDict()
        for i in range(len(cells)):
            buckets.setdefault(tuple(int(x) for x in cells[i]), []).append(i)
        keys = sorted(buckets)
        order, bi = [], 0
        while len(order) < n and any(buckets[k] for k in keys):
            k = keys[bi % len(keys)]
            if buckets[k]:
                order.append(buckets[k].pop(0))
            bi += 1
        return np.array(order[:n])

    def prepare_targets(self):
        d = Path(to_absolute_path(self.rigid["data_path"])) / self.rigid["split"]
        start = torch.load(d / "start_states.pth").numpy().astype(np.float64)   # (N,5)
        goal = torch.load(d / "goal_states.pth").numpy().astype(np.float64)     # (N,5)
        cells = torch.load(d / "region_cells.pth").numpy()                      # (N,2)
        n = self.n_evals
        if self.rigid.get("stratify_regions", True):
            order = self._stratified_order(cells, n)
        else:
            order = np.arange(min(n, len(start)))
        assert len(order) >= n, f"only {len(order)} goals available, need n_evals={n}"
        start, goal, self.region_cells = start[order], goal[order], cells[order]
        # env.prepare expects 7-dim states (with velocity); rigid states are 5-dim -> pad 0
        init7 = np.concatenate([start, np.zeros((len(start), 2))], axis=1)
        goal7 = np.concatenate([goal, np.zeros((len(goal), 2))], axis=1)
        obs_0, state_0 = self.env.prepare(self.eval_seed, init7)
        obs_g, state_g = self.env.prepare(self.eval_seed, goal7)
        self.obs_0 = {k: np.expand_dims(v, axis=1) for k, v in obs_0.items()}
        self.obs_g = {k: np.expand_dims(v, axis=1) for k, v in obs_g.items()}
        self.state_0 = state_0
        self.state_g = state_g
        self.gt_actions = None
        # goal-frame pusher = the transformed contact pose (cols 0:2 of goal_state); the
        # rollout pusher also ends near contact -> mask the same patches both sides.
        sg = np.asarray(state_g, dtype=np.float64)
        self.goal_pusher_xy = sg[:, 0:2].copy()
        self.real_pusher_xy = sg[:, 0:2].copy()
        rc = Counter(tuple(int(x) for x in c) for c in self.region_cells)
        print("[rigid] %d goals | regions: %s" % (
            n, ", ".join(f"{region_name(k)}:{v}" for k, v in sorted(rc.items()))))

    def _attach_goal_override(self):
        gs = self.cfg_dict["goal_source"]
        target = (self.planner.sub_planner
                  if isinstance(self.planner, MPCPlanner) else self.planner)
        trans_obs_g = move_to_device(self.data_preprocessor.transform_obs(self.obs_g), self.device)
        with torch.no_grad():
            zg = self.wm.encode_obs(trans_obs_g)
        if gs == "random":
            zg["visual"] = torch.roll(zg["visual"], shifts=1, dims=0)
            target.z_obs_g_override = zg
            print("[rigid] random floor: rolled goal latents (unrelated to each spec)")
            return
        raise NotImplementedError(f"goal_source={gs!r} (bridge/swapped_spec) is wired in Phase 4")


def planning_main_rigid(cfg_dict):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

    env = SubprocVectorEnv([
        (lambda: gym.make("pusht", with_velocity=True, with_target=False,
                          success_mode="regional"))
        for _ in range(cfg_dict["n_evals"])
    ])

    ws = RigidPlanWorkspace(cfg_dict, model, dset, env, model_cfg.frameskip, wandb_run=None)
    logs = ws.perform_planning()

    # per-region SR (from the MPC per-eval success), aligned to region_cells
    succ = ws.planner.is_success if isinstance(ws.planner, MPCPlanner) else None
    if succ is not None and ws.region_cells is not None:
        per = defaultdict(lambda: [0, 0])
        for c, s in zip(ws.region_cells, succ):
            k = tuple(int(x) for x in c)
            per[k][0] += int(bool(s)); per[k][1] += 1
        print("\n=== per-region SR (retrained clean dynamics, oracle) ===")
        for k in sorted(per):
            ns, nt = per[k]
            print(f"  {region_name(k):16s} {ns}/{nt} = {ns/nt:.2f}")

    headline = {k: v for k, v in logs.items() if ("success" in k or "dist" in k)}
    print(f"\n=== RIGID {cfg_dict['goal_source'].upper()} RESULT (n={cfg_dict['n_evals']}, "
          f"model={cfg_dict['model_name']}, masked={cfg_dict.get('mask_pusher')}, "
          f"alpha={(cfg_dict.get('objective') or {}).get('alpha')}) ===")
    print(headline)
    return logs


@hydra.main(config_path="conf", config_name="plan_pusht_rigid", version_base=None)
def main(cfg: OmegaConf):
    with open_dict(cfg):
        cfg["saved_folder"] = os.getcwd()
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_dict["wandb_logging"] = False
    planning_main_rigid(cfg_dict)


if __name__ == "__main__":
    main()

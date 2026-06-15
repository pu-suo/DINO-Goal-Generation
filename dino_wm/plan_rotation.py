"""Option B Part 4: oracle pose-gen gate for rotation-command goals (raw clean
sub-segments, REUSED clean_retrain dynamics).

The deployable oracle plans toward enc(real clean goal frame) under the masked
energy (objective.alpha=0 + mask_pusher), scored by the Option-B metric: position
point-tolerance (20px to the scene-determined goal block position) AND rotation
BUCKET membership (achieved relative rotation in the commanded sign+band). Reports
position-success and rotation-success SEPARATELY (the rotation axis is the load-
bearing one; pos is scene-determined) and per command bucket. This is the CEILING
g inherits -- STOP and report before training g.

No rigid transform, no retrain: raw clean sub-segments are in clean_retrain's
training distribution (clean pusht_noise trajectory windows; block-TF 7.06), and
Part D showed rotation is not leaked from raw z_start.

Box (4090):
  DATASET_DIR=/workspace/data /workspace/envs/dino_wm/bin/python plan_rotation.py \
    model_name=clean_retrain ckpt_base_path=/workspace/dino_goal/dino_wm n_evals=40 \
    goal_source=oracle mask_pusher=true objective.alpha=0 plan_eval_mode=true \
    fast_tf32=true fast_sdpa=true planner.sub_planner.traj_chunk=3 \
    planner.sub_planner.num_samples=300 planner.sub_planner.opt_steps=30 planner.max_iter=10
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
from utils import seed
from plan import PlanWorkspace, load_model, apply_fast_flags
from datasets.rotation_command import all_buckets, bucket_name
from metrics.rotation_goal_success import rotation_goal_success

warnings.filterwarnings("ignore")


class RotationPlanWorkspace(PlanWorkspace):
    def __init__(self, cfg_dict, wm, dset, env, frameskip, wandb_run):
        self.subseg = cfg_dict["subseg"]
        self.rot_bucket = None
        super().__init__(cfg_dict=cfg_dict, wm=wm, dset=dset, env=env,
                         env_name="pusht", frameskip=frameskip, wandb_run=wandb_run)

    def _stratified_order(self, buckets, n):
        """Round-robin across the (sign,band) buckets so each command is represented."""
        groups = OrderedDict()
        for i in range(len(buckets)):
            groups.setdefault(tuple(int(x) for x in buckets[i]), []).append(i)
        keys = [b for b in all_buckets() if b in groups and groups[b]]
        order, bi = [], 0
        while len(order) < n and any(groups[k] for k in keys):
            k = keys[bi % len(keys)]
            if groups[k]:
                order.append(groups[k].pop(0))
            bi += 1
        return np.array(order[:n])

    def prepare_targets(self):
        d = Path(to_absolute_path(self.subseg["data_path"])) / self.subseg["split"]
        start = torch.load(d / "start_states.pth").numpy().astype(np.float64)   # (N,5)
        goal = torch.load(d / "goal_states.pth").numpy().astype(np.float64)
        buckets = torch.load(d / "rot_buckets.pth").numpy()                     # (N,2)
        n = self.n_evals
        order = (self._stratified_order(buckets, n) if self.subseg.get("stratify", True)
                 else np.arange(min(n, len(start))))
        assert len(order) >= n, f"only {len(order)} goals available, need n_evals={n}"
        start, goal, self.rot_bucket = start[order], goal[order], buckets[order]
        init7 = np.concatenate([start, np.zeros((len(start), 2))], axis=1)
        goal7 = np.concatenate([goal, np.zeros((len(goal), 2))], axis=1)
        obs_0, state_0 = self.env.prepare(self.eval_seed, init7)
        obs_g, state_g = self.env.prepare(self.eval_seed, goal7)
        self.obs_0 = {k: np.expand_dims(v, axis=1) for k, v in obs_0.items()}
        self.obs_g = {k: np.expand_dims(v, axis=1) for k, v in obs_g.items()}
        self.state_0 = state_0
        self.state_g = state_g
        self.gt_actions = None
        sg = np.asarray(state_g, dtype=np.float64)
        self.goal_pusher_xy = sg[:, 0:2].copy()
        self.real_pusher_xy = sg[:, 0:2].copy()
        # rotation metric needs the REALIZED start angle + goal block position
        self.start_angle = np.asarray(state_0, dtype=np.float64)[:, 4].copy()
        self.goal_block_xy = sg[:, 2:4].copy()
        bc = Counter(tuple(int(x) for x in b) for b in self.rot_bucket)
        print("[rot] %d goals | buckets: %s" % (
            n, ", ".join(f"{bucket_name(k)}:{v}" for k, v in sorted(bc.items()))))

    def perform_planning(self):
        actions, action_len = self.planner.plan(obs_0=self.obs_0, obs_g=self.obs_g, actions=None)
        # plot=False: skip the cosmetic VQ-VAE decode (it OOM'd the last oracle); metrics unchanged
        logs, _, _, e_states = self.evaluator.eval_actions(
            actions.detach(), action_len, save_video=False, plot=False)
        e_states = np.asarray(e_states)                       # (n, T+1, d)
        e_final = e_states[:, -1, :]
        res = [rotation_goal_success(self.goal_block_xy[i], self.start_angle[i],
                                     e_final[i], tuple(int(x) for x in self.rot_bucket[i]))
               for i in range(self.n_evals)]
        pos = np.array([r["pos_ok"] for r in res], dtype=float)
        rot = np.array([r["rot_ok"] for r in res], dtype=float)
        succ = np.array([r["success"] for r in res], dtype=float)
        self.is_success = succ.astype(bool)

        print(f"\n=== ROTATION ORACLE (n={self.n_evals}, model={self.cfg_dict['model_name']}, "
              f"masked={self.cfg_dict.get('mask_pusher')}, alpha={(self.cfg_dict.get('objective') or {}).get('alpha')}) ===")
        print(f"  SUCCESS(pos&rot) = {succ.mean():.3f}   pos_ok = {pos.mean():.3f}   rot_ok = {rot.mean():.3f}")
        per = defaultdict(lambda: [0, 0, 0, 0])               # succ,pos,rot,total
        for b, r in zip(self.rot_bucket, res):
            k = tuple(int(x) for x in b)
            per[k][0] += int(r["success"]); per[k][1] += int(r["pos_ok"])
            per[k][2] += int(r["rot_ok"]); per[k][3] += 1
        print("  per-bucket  (succ / pos / rot / n):")
        for k in sorted(per):
            s, p, ro, t = per[k]
            print(f"    {bucket_name(k):14s}  {s}/{t}  (pos {p}/{t}, rot {ro}/{t})")
        return {"final_eval/success_rate": float(succ.mean()),
                "final_eval/pos_ok": float(pos.mean()),
                "final_eval/rot_ok": float(rot.mean())}


def planning_main(cfg_dict):
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
        model.eval(); print("[plan] plan_eval_mode=True -> model.eval()")
    apply_fast_flags(cfg_dict, model)

    env = SubprocVectorEnv([
        (lambda: gym.make("pusht", with_velocity=True, with_target=False))
        for _ in range(cfg_dict["n_evals"])
    ])
    ws = RotationPlanWorkspace(cfg_dict, model, dset, env, model_cfg.frameskip, wandb_run=None)
    return ws.perform_planning()


@hydra.main(config_path="conf", config_name="plan_pusht_rotation", version_base=None)
def main(cfg: OmegaConf):
    with open_dict(cfg):
        cfg["saved_folder"] = os.getcwd()
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_dict["wandb_logging"] = False
    planning_main(cfg_dict)


if __name__ == "__main__":
    main()

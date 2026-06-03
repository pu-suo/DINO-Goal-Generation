"""
Phase 0.5 - oracle ceiling for multi-color PushT.

Plans with CEM toward z_g = enc(o_goal), where o_goal is the REAL rendered frame
with the block at the NAMED target (the oracle sees ground truth). Uses the
SHIPPED PushT dynamics (frozen) inside the multi-color env, so this is the
ceiling `g` is measured against in Phase 1.

It reuses plan.py's PlanWorkspace machinery and only overrides target prep:
sample decorrelated, split-aware layouts -> install the N decals on each env
worker -> obs_0 = render(init), obs_g = render(block @ named target). Optionally
drops the pusher patches from the CEM energy (manipulator-masked).

    cd dino_wm
    DATASET_DIR=/data python plan_multicolor.py \
        model_name=pusht ckpt_base_path=/ckpts n_evals=50 \
        multicolor.combo_split=heldout                      # HEADLINE oracle
    # masked-energy variant:
    DATASET_DIR=/data python plan_multicolor.py multicolor.combo_split=heldout \
        multicolor.use_manipulator_mask=true objective.alpha=0 multicolor.mask_tag=_masked
"""
import os
import warnings

import gym
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from pathlib import Path

import custom_resolvers  # noqa: F401  (registers ${replace_slash})
from env.venv import SubprocVectorEnv
from utils import seed
from planning.mpc import MPCPlanner
from env.pusht.multicolor_common import pusher_patch_mask
from env.pusht import multicolor_sampler as mcs
from plan import PlanWorkspace, load_model

warnings.filterwarnings("ignore")


class MultiColorPlanWorkspace(PlanWorkspace):
    def __init__(self, cfg_dict, wm, dset, env, frameskip, wandb_run):
        self.mc = cfg_dict["multicolor"]
        super().__init__(
            cfg_dict=cfg_dict, wm=wm, dset=dset, env=env,
            env_name="pusht_multicolor", frameskip=frameskip, wandb_run=wandb_run,
        )
        if self.mc["use_manipulator_mask"]:
            masks = np.stack([pusher_patch_mask(self.state_g[i, :2]) for i in range(self.n_evals)])
            masks = torch.tensor(masks, device=self.device)
            target = self.planner.sub_planner if isinstance(self.planner, MPCPlanner) else self.planner
            target.patch_mask = masks
            print(f"[mask] dropped {int((masks[0] == 0).sum())} pusher patches/eval from the energy")

    def prepare_targets(self):
        mc = self.mc
        train_combos, test_combos = mcs.make_combo_split(
            mc["n_targets"], mc["n_bins"], mc["heldout_frac"], mc["split_seed"])
        if mc["combo_split"] == "heldout":
            allowed, active = train_combos, test_combos
        elif mc["combo_split"] == "train":
            allowed, active = train_combos, None
        else:
            allowed, active = None, None

        layouts = [
            mcs.sample_layout(mc["layout_seed_base"] + i, n_targets=mc["n_targets"],
                              with_velocity=True, n_bins=mc["n_bins"],
                              allowed_combos=allowed, active_combos=active,
                              max_goal_dist=mc.get("max_goal_dist"),
                              max_goal_angle=mc.get("max_goal_angle"))
            for i in range(self.n_evals)
        ]
        self.layouts = layouts
        self.instructions = [l["instruction"] for l in layouts]
        self.active_colors = [l["active_color"] for l in layouts]
        print(f"[oracle] {self.n_evals} evals | combo_split={mc['combo_split']} | "
              f"example: \"{layouts[0]['instruction']}\"")

        # install the N decals on each env worker (via the update_env dispatch)
        self.env.update_env(layouts)
        init_states = np.stack([l["init_state"] for l in layouts])
        goal_states = init_states.copy()
        for i, l in enumerate(layouts):
            goal_states[i, 2:5] = l["goal_pose"]  # block -> named target

        obs_0, state_0 = self.env.prepare(self.eval_seed, init_states)
        obs_g, state_g = self.env.prepare(self.eval_seed, goal_states)
        self.obs_0 = {k: np.expand_dims(v, axis=1) for k, v in obs_0.items()}
        self.obs_g = {k: np.expand_dims(v, axis=1) for k, v in obs_g.items()}
        self.state_0 = state_0
        self.state_g = state_g
        self.gt_actions = None


def planning_main_mc(cfg_dict):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mc = cfg_dict["multicolor"]
    model_path = f"{cfg_dict['ckpt_base_path']}/outputs/{cfg_dict['model_name']}/"
    model_cfg = OmegaConf.load(os.path.join(model_path, "hydra.yaml"))
    seed(cfg_dict["seed"])

    # dataset only provides normalization stats + transform. 'model' -> pusht_noise
    # stats (correct for the SHIPPED dynamics); 'multicolor' -> for a retrained model.
    if mc["stats_source"] == "model":
        _, dset = hydra.utils.call(model_cfg.env.dataset, num_hist=model_cfg.num_hist,
                                   num_pred=model_cfg.num_pred, frameskip=model_cfg.frameskip)
    else:
        from datasets.pusht_multicolor_dset import load_pusht_multicolor_slice_train_val
        from datasets.img_transforms import default_transform
        _, dset = load_pusht_multicolor_slice_train_val(
            transform=default_transform(model_cfg.img_size), data_path=mc["data_path"],
            num_hist=model_cfg.num_hist, num_pred=model_cfg.num_pred, frameskip=model_cfg.frameskip)
    dset = dset["valid"]

    model_ckpt = Path(model_path) / "checkpoints" / f"model_{cfg_dict['model_epoch']}.pth"
    model = load_model(model_ckpt, model_cfg, model_cfg.num_action_repeat, device=device)

    env_kwargs = dict(with_velocity=True, with_target=True, n_targets=mc["n_targets"],
                      outline_thickness=mc["outline_thickness"],
                      success_threshold=mc["success_threshold"], n_bins=mc["n_bins"])
    env = SubprocVectorEnv(
        [lambda: gym.make("pusht_multicolor", **env_kwargs) for _ in range(cfg_dict["n_evals"])]
    )

    ws = MultiColorPlanWorkspace(cfg_dict, model, dset, env, model_cfg.frameskip, wandb_run=None)
    logs = ws.perform_planning()
    headline = {k: v for k, v in logs.items() if any(s in k for s in ("success", "coverage"))}
    print("\n=== ORACLE RESULT ===")
    print(f"combo_split={mc['combo_split']} masked={mc['use_manipulator_mask']}")
    print(headline)
    return logs


@hydra.main(config_path="conf", config_name="plan_pusht_multicolor", version_base=None)
def main(cfg: OmegaConf):
    with open_dict(cfg):
        cfg["saved_folder"] = os.getcwd()
    # resolve=True so ${oc.env:DATASET_DIR} etc. are concrete (cfg_to_dict leaves
    # interpolations unresolved); our config has no top-level list values.
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_dict["wandb_logging"] = False
    planning_main_mc(cfg_dict)


if __name__ == "__main__":
    main()

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
from utils import seed, move_to_device
from planning.mpc import MPCPlanner
from env.pusht.multicolor_common import pusher_patch_mask, contact_pusher_pose
from env.pusht import multicolor_sampler as mcs
from plan import PlanWorkspace, load_model, apply_fast_flags

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
        if cfg_dict["goal_source"] == "bridge":
            # Stage-2: replace the planner's goal latent with g(z_start, instruction).
            # Must run AFTER the planner exists (same post-construction pattern as the
            # manipulator mask above).
            self._attach_bridge_goal()

    def _sample_oracle_layouts(self):
        """Decorrelated, split-aware layouts + init/goal states (block -> named target).
        Shared by the named_target oracle and the Stage-2 bridge path; the goal-frame
        PUSHER placement stays with each caller."""
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
        print(f"[{self.goal_source}] {self.n_evals} evals | combo_split={mc['combo_split']} | "
              f"example: \"{layouts[0]['instruction']}\"")

        # install the N decals on each env worker (via the update_env dispatch)
        self.env.update_env(layouts)
        init_states = np.stack([l["init_state"] for l in layouts])
        goal_states = init_states.copy()
        for i, l in enumerate(layouts):
            goal_states[i, 2:5] = l["goal_pose"]  # block -> named target
        return layouts, init_states, goal_states

    def prepare_targets(self):
        # Real-goal CONTROL. When goal_source != "named_target", plan toward a REAL
        # reachable multicolor goal -- a real trajectory's endpoint, so block AND
        # pusher sit in a physically-consistent pose -- with the standard alpha=1
        # energy (PlanWorkspace's dset/random_state path). This isolates the
        # dynamics model from the FABRICATED named-target goal:
        #   SR_control >> SR_named  => the cap is our goal construction (pusher
        #                              placement), model is fine -> fix the oracle.
        #   SR_control ~ SR_named   => the model itself is the limit -> rollout-aware
        #                              retraining is the real lever.
        if self.goal_source == "bridge":
            return self._prepare_targets_bridge()
        if self.goal_source != "named_target":
            return super().prepare_targets()
        mc = self.mc
        layouts, init_states, goal_states = self._sample_oracle_layouts()

        # Where to put the pusher in the oracle goal. The pusher's true goal-time
        # pose is unknown, and the choice interacts with objective.alpha (the
        # proprio/manipulator energy term):
        #   "hide"   -> pusher off-screen; REQUIRES alpha=0. The energy then gives
        #               NO signal pulling the pusher toward the block (sparse reward
        #               -> CEM only solves the configs it stumbles onto). Stalls ~0.3.
        #   "behind" -> put the pusher where a real push ENDS: just behind the block
        #               (at the target), on the side it was pushed FROM. With alpha>0
        #               this restores the dense pusher-approach guidance that makes
        #               stock PushT planning work (~0.9), without the start-pin bug
        #               (the start-pin made the energy fight the push). The goal is
        #               then a plausible, on-manifold real state (block@target +
        #               pusher behind it), close to stock's real-goal setup.
        pusher_mode = mc.get("goal_pusher", "hide")
        if pusher_mode == "behind":
            offset = float(mc.get("goal_pusher_offset", 40.0))
            for i in range(len(layouts)):
                d = goal_states[i, 2:4] - init_states[i, 2:4]  # block start -> target
                n = float(np.linalg.norm(d))
                if n > 1e-3:
                    goal_states[i, 0:2] = goal_states[i, 2:4] - (d / n) * offset
                # else: ~no translation needed; keep the start pusher pose
        elif pusher_mode == "contact":
            # Like "behind" but on the REAL contact surface: accounts for the block
            # heading theta so the pusher is just outside the rotated T and never
            # overlaps it (naive "behind" overlaps the T for ~62% of rotated goals,
            # an impossible goal latent). Pair with objective.alpha=1.
            for i in range(len(layouts)):
                p = contact_pusher_pose(init_states[i, 2:4], goal_states[i, 2:5])
                if p is not None:
                    goal_states[i, 0:2] = p
                # else: ~no translation needed; keep the start pusher pose
        elif pusher_mode == "hide":
            goal_states[:, 0] = -1000.0
            goal_states[:, 1] = -1000.0

        obs_0, state_0 = self.env.prepare(self.eval_seed, init_states)
        obs_g, state_g = self.env.prepare(self.eval_seed, goal_states)
        self.obs_0 = {k: np.expand_dims(v, axis=1) for k, v in obs_0.items()}
        self.obs_g = {k: np.expand_dims(v, axis=1) for k, v in obs_g.items()}
        self.state_0 = state_0
        self.state_g = state_g
        self.gt_actions = None

    def _prepare_targets_bridge(self):
        """Stage-2: the goal is g(z_start, instruction) -- no goal image exists.

        Same decorrelated layouts/success target as the named_target oracle, but the
        goal-frame pusher stays at its START pose: that is g's training convention
        (the cached goal frames keep the pusher at init), so it is both where g
        leaves the pusher in the synthesized latent and the right patches for the
        manipulator mask to drop. obs_g is rendered ONLY as (a) the evaluator's plot
        reference and (b) the proprio entry of the goal dict; the planner's VISUAL
        target is replaced by g's output in _attach_bridge_goal (z_obs_g_override).
        Success stays the env pose check of the block vs the NAMED target.
        """
        _, init_states, goal_states = self._sample_oracle_layouts()
        # goal_states pusher cols [0:2] are already the start pusher (copied from
        # init_states); only the block was moved to the named target.
        obs_0, state_0 = self.env.prepare(self.eval_seed, init_states)
        obs_g, state_g = self.env.prepare(self.eval_seed, goal_states)
        self.obs_0 = {k: np.expand_dims(v, axis=1) for k, v in obs_0.items()}
        self.obs_g = {k: np.expand_dims(v, axis=1) for k, v in obs_g.items()}
        self.state_0 = state_0
        self.state_g = state_g
        self.gt_actions = None

    def _attach_bridge_goal(self):
        """Compute z_goal = g(z_start, instruction) and override the CEM goal latent.

        Runs after the planner exists. The override dict mirrors wm.encode_obs's
        output: visual <- g's synthesized grid; proprio <- the rendered placeholder's
        encoding (inert under the deployable alpha=0 energy, shape-consistent
        otherwise). Loud failure beats silent oracle: if anything here breaks we
        raise, because falling back to enc(obs_g) would silently run the ORACLE."""
        from hydra.utils import to_absolute_path
        from models.bridge import BridgeG, FrozenTextEncoder

        ckpt_rel = self.mc.get("bridge_ckpt")
        if not ckpt_rel:
            raise ValueError("goal_source=bridge requires multicolor.bridge_ckpt")
        ckpt_path = to_absolute_path(ckpt_rel)  # resolved vs the LAUNCH cwd (hydra original dir)
        ck = torch.load(ckpt_path, map_location=self.device)
        c = ck["config"]
        g = BridgeG(dim=c["dim"], depth=c["depth"], heads=c["heads"], d_text=c["d_text"]).to(self.device)
        g.load_state_dict(ck["state_dict"])
        g.eval()
        # text_model=None means the ckpt was trained with --dummy_text (deterministic random
        # tokens): real MiniLM tokens are shape-compatible but statistically unrelated, so
        # planning would silently produce garbage goals. Refuse. (A genuinely MISSING key =
        # legacy ckpt -> MiniLM default, stated loudly.)
        if "text_model" in ck and ck["text_model"] is None:
            raise ValueError(f"{ckpt_path} was trained with --dummy_text (text_model=None); "
                             f"refusing to plan with a real text encoder")
        text_model = ck.get("text_model", "sentence-transformers/all-MiniLM-L6-v2")
        text_enc = FrozenTextEncoder(text_model, max_len=c.get("text_max_len", 16), device="cpu")
        print(f"[bridge] text encoder: {text_model} (max_len={c.get('text_max_len', 16)})")
        tok, tmask = text_enc(self.instructions)

        trans_obs_0 = move_to_device(self.data_preprocessor.transform_obs(self.obs_0), self.device)
        trans_obs_g = move_to_device(self.data_preprocessor.transform_obs(self.obs_g), self.device)
        with torch.no_grad():
            z0 = self.wm.encode_obs(trans_obs_0)
            zg = self.wm.encode_obs(trans_obs_g)          # placeholder: proprio + plots
            z_goal_vis = g(z0["visual"][:, 0],            # (B,196,384)
                           tok.to(self.device), tmask.to(self.device))
        ref_cos = torch.nn.functional.cosine_similarity(
            z_goal_vis.reshape(self.n_evals, -1),
            zg["visual"][:, 0].reshape(self.n_evals, -1), dim=-1)
        zg["visual"] = z_goal_vis.unsqueeze(1).to(zg["visual"].dtype)

        target = self.planner.sub_planner if isinstance(self.planner, MPCPlanner) else self.planner
        if not hasattr(target, "z_obs_g_override"):
            # e.g. the GD planner: it would silently encode the placeholder obs_g and
            # run the ORACLE while reporting bridge results.
            raise TypeError(f"goal_source=bridge requires a planner that consumes "
                            f"z_obs_g_override (CEMPlanner declares it in __init__); got "
                            f"{type(target).__name__}")
        target.z_obs_g_override = zg
        cost_cfg = self.cfg_dict.get("cost") or self.cfg_dict.get("objective") or {}
        alpha = cost_cfg.get("alpha")
        print(f"[bridge] z_obs_g_override attached from {ckpt_path} | "
              f"cos(g_goal, rendered-ref) mean={ref_cos.mean():.3f} min={ref_cos.min():.3f}")
        if alpha is not None and alpha not in (0, 0.0):
            print(f"[bridge] WARN: alpha={alpha} != 0 scores the proprio term against a "
                  f"fabricated start-pose pusher; the deployable energy is alpha=0 + "
                  f"multicolor.use_manipulator_mask=true")


def planning_main_mc(cfg_dict):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mc = cfg_dict["multicolor"]
    # Fail fast on a missing bridge ckpt BEFORE spinning up envs / loading the model.
    if cfg_dict["goal_source"] == "bridge":
        from hydra.utils import to_absolute_path
        _p = mc.get("bridge_ckpt")
        if not _p or not os.path.exists(to_absolute_path(_p)):
            raise FileNotFoundError(
                f"goal_source=bridge requires an existing multicolor.bridge_ckpt (got {_p!r})")
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
    # plan_eval_mode: force eval() at plan time. Expected NO-OP -- ckpts are pickled
    # post-val() in eval mode (see the corrected comment in plan.py and
    # scripts/check_ckpt_train_mode.py). Kept as belt-and-braces fast-config hygiene.
    if cfg_dict.get("plan_eval_mode", False):
        model.eval()
        print("[plan] plan_eval_mode=True -> model.eval() (expected no-op: ckpts are "
              "saved post-val() in eval mode; see scripts/check_ckpt_train_mode.py)")

    apply_fast_flags(cfg_dict, model)

    env_kwargs = dict(with_velocity=True, with_target=True, n_targets=mc["n_targets"],
                      outline_thickness=mc["outline_thickness"],
                      success_threshold=mc["success_threshold"], n_bins=mc["n_bins"])
    env = SubprocVectorEnv(
        [lambda: gym.make("pusht_multicolor", **env_kwargs) for _ in range(cfg_dict["n_evals"])]
    )

    ws = MultiColorPlanWorkspace(cfg_dict, model, dset, env, model_cfg.frameskip, wandb_run=None)
    logs = ws.perform_planning()
    headline = {k: v for k, v in logs.items() if any(s in k for s in ("success", "coverage"))}
    print(f"\n=== {cfg_dict['goal_source'].upper()} RESULT ===")
    print(f"goal_source={cfg_dict['goal_source']} combo_split={mc['combo_split']} "
          f"masked={mc['use_manipulator_mask']}")
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

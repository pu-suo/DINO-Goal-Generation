import torch
import numpy as np
from einops import rearrange, repeat
from .base_planner import BasePlanner
from utils import move_to_device


class CEMPlanner(BasePlanner):
    def __init__(
        self,
        horizon,
        topk,
        num_samples,
        var_scale,
        opt_steps,
        eval_every,
        wm,
        action_dim,
        objective_fn,
        preprocessor,
        evaluator,
        wandb_run,
        logging_prefix="plan_0",
        log_filename="logs.json",
        **kwargs,
    ):
        super().__init__(
            wm,
            action_dim,
            objective_fn,
            preprocessor,
            evaluator,
            wandb_run,
            log_filename,
        )
        self.horizon = horizon
        self.topk = topk
        self.num_samples = num_samples
        self.var_scale = var_scale
        self.opt_steps = opt_steps
        self.eval_every = eval_every
        self.logging_prefix = logging_prefix
        # optional (n_evals, P) manipulator-masked energy; set by the multi-color
        # planning workspace. None -> stock full-grid energy (unchanged behavior).
        self.patch_mask = None
        # Speed knob (default ON, result-preserving): encode the start obs ONCE and
        # roll out from the cached latent instead of re-encoding it for every CEM
        # candidate every opt-step. The frozen encoder is RNG-free, so this does not
        # perturb the predictor's dropout RNG stream -> scores are identical to FP
        # tolerance. Set fast_encode=false (e.g. +planner.sub_planner.fast_encode=false)
        # to fall back to the original per-candidate re-encode path for A/B regression.
        self.fast_encode = bool(kwargs.get("fast_encode", True))

    def init_mu_sigma(self, obs_0, actions=None):
        """
        actions: (B, T, action_dim) torch.Tensor, T <= self.horizon
        mu, sigma could depend on current obs, but obs_0 is only used for providing n_evals for now
        """
        n_evals = obs_0["visual"].shape[0]
        sigma = self.var_scale * torch.ones([n_evals, self.horizon, self.action_dim])
        if actions is None:
            mu = torch.zeros(n_evals, 0, self.action_dim)
        else:
            mu = actions
        device = mu.device
        t = mu.shape[1]
        remaining_t = self.horizon - t

        if remaining_t > 0:
            new_mu = torch.zeros(n_evals, remaining_t, self.action_dim)
            mu = torch.cat([mu, new_mu.to(device)], dim=1)
        return mu, sigma

    def plan(self, obs_0, obs_g, actions=None):
        """
        Args:
            actions: normalized
        Returns:
            actions: (B, T, action_dim) torch.Tensor, T <= self.horizon
        """
        trans_obs_0 = move_to_device(
            self.preprocessor.transform_obs(obs_0), self.device
        )
        trans_obs_g = move_to_device(
            self.preprocessor.transform_obs(obs_g), self.device
        )
        # Encode the goal AND the start observation ONCE per plan() (the goal was
        # already cached; the start is the speed fix). Both encoders are frozen and
        # RNG-free, so this is identical to encoding inside the loop -- it just stops
        # the inner loop from re-running DINOv2 on the same frame for every candidate.
        with torch.no_grad():
            z_obs_g = self.wm.encode_obs(trans_obs_g)
            z_obs_0 = self.wm.encode_obs(trans_obs_0) if self.fast_encode else None

        mu, sigma = self.init_mu_sigma(obs_0, actions)
        mu, sigma = mu.to(self.device), sigma.to(self.device)
        n_evals = mu.shape[0]

        for i in range(self.opt_steps):
            # optimize individual instances
            losses = []
            for traj in range(n_evals):
                cur_z_obs_g = {
                    key: repeat(
                        arr[traj].unsqueeze(0), "1 ... -> n ...", n=self.num_samples
                    )
                    for key, arr in z_obs_g.items()
                }
                action = (
                    torch.randn(self.num_samples, self.horizon, self.action_dim).to(
                        self.device
                    )
                    * sigma[traj]
                    + mu[traj]
                )
                action[0] = mu[traj]  # optional: make the first one mu itself
                with torch.no_grad():
                    if self.fast_encode:
                        # Speed path: roll out from the cached start latent, broadcast
                        # to num_samples. No DINOv2 re-encode -> no extra RNG draws.
                        cur_z_obs_0 = {
                            key: repeat(
                                arr[traj].unsqueeze(0), "1 ... -> n ...", n=self.num_samples
                            )
                            for key, arr in z_obs_0.items()
                        }
                        i_z_obses, i_zs = self.wm.rollout_from_zobs(
                            z_obs_0=cur_z_obs_0,
                            act=action,
                        )
                    else:
                        # Original path (A/B baseline): re-encode the start obs for
                        # every candidate inside the inner loop.
                        cur_trans_obs_0 = {
                            key: repeat(
                                arr[traj].unsqueeze(0), "1 ... -> n ...", n=self.num_samples
                            )
                            for key, arr in trans_obs_0.items()
                        }
                        i_z_obses, i_zs = self.wm.rollout(
                            obs_0=cur_trans_obs_0,
                            act=action,
                        )

                vis_mask = None if self.patch_mask is None else self.patch_mask[traj]
                loss = self.objective_fn(i_z_obses, cur_z_obs_g, vis_mask=vis_mask)
                topk_idx = torch.argsort(loss)[: self.topk]
                topk_action = action[topk_idx]
                losses.append(loss[topk_idx[0]].item())
                mu[traj] = topk_action.mean(dim=0)
                sigma[traj] = topk_action.std(dim=0)

            self.wandb_run.log(
                {f"{self.logging_prefix}/loss": np.mean(losses), "step": i + 1}
            )
            if self.evaluator is not None and i % self.eval_every == 0:
                # plot=False: the per-opt-step debug image (VQ-VAE decode + PNG dump)
                # is pure visualization. Skipping it for inner evals does not touch
                # `successes`/`logs` (computed before plotting) nor the RNG stream (the
                # decoder is RNG-free), so the early-break and SR are unchanged.
                logs, successes, _, _ = self.evaluator.eval_actions(
                    mu, filename=f"{self.logging_prefix}_output_{i+1}", plot=False
                )
                logs = {f"{self.logging_prefix}/{k}": v for k, v in logs.items()}
                logs.update({"step": i + 1})
                self.wandb_run.log(logs)
                self.dump_logs(logs)
                if np.all(successes):
                    break  # terminate planning if all success

        return mu, np.full(n_evals, np.inf)  # all actions are valid

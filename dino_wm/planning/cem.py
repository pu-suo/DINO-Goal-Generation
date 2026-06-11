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
        # --- Fast-config knobs (docs/PLANNING_SPEED_PROFILE.md "Fast config bundle").
        # Both default to the stock behavior; both are result-CHANGING when enabled and
        # are covered by the single bundled re-anchor run, NOT by per-flag A/Bs.
        # skip_succeeded: skip rollout+scoring+refit for trajectories the MPC outer loop
        # has already marked successful (their taken actions are zeroed by MPC anyway).
        # The per-traj candidate randn draw is STILL executed for skipped trajs so the
        # CPU RNG stream seen by the remaining trajectories is byte-identical.
        self.skip_succeeded = bool(kwargs.get("skip_succeeded", False))
        # success_mask: (n_evals,) bool, set per MPC iter by MPCPlanner when
        # skip_succeeded is on. None -> all trajectories active.
        self.success_mask = None
        # traj_chunk: number of trajectories whose num_samples-candidate rollouts are
        # batched into ONE rollout_from_zobs call per opt step (1 = stock sequential
        # loop, byte-identical code path). Candidates per call = traj_chunk*num_samples;
        # pair with fast_sdpa (naive attention would materialize a (B,h,T,T) score
        # tensor: ~22 MB/candidate fp32 at 588 tokens -> OOM well before 1k candidates).
        self.traj_chunk = int(kwargs.get("traj_chunk", 1))
        if (self.traj_chunk > 1 or self.skip_succeeded) and not self.fast_encode:
            raise ValueError("traj_chunk>1 / skip_succeeded=true require fast_encode=true "
                             "(the fast branch rolls out from the cached z_obs_0).")

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

        # Fast-config branch (skip_succeeded / traj_chunk>1). The stock branch below is
        # untouched (byte-identical) when both knobs are at their defaults.
        if (self.skip_succeeded and self.success_mask is not None) or self.traj_chunk > 1:
            return self._plan_fast(mu, sigma, z_obs_0, z_obs_g, n_evals)

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

    def _plan_fast(self, mu, sigma, z_obs_0, z_obs_g, n_evals):
        """Fast-config CEM step loop: skip already-succeeded trajectories and/or batch
        traj_chunk trajectories' candidate rollouts into one GPU call.

        Equivalence to the stock loop (up to floating-point reduction order):
        - Candidate draws: one torch.randn(num_samples, horizon, action_dim) per traj,
          in traj order, EVERY opt step -- including for skipped trajs (drawn and
          discarded) -- so the CPU RNG stream is byte-identical to the stock loop and
          active trajectories receive exactly the candidates they would have received.
          (Within an opt step, traj t's draw reads only mu[t]/sigma[t], which the stock
          loop updates only after t's own draw -- so drawing up front is equivalent.)
        - Scoring/refit: per-traj objective + topk + mu/sigma update, identical math on
          slices of the batched rollout. Batching changes only cuBLAS reduction order
          (NOT result-preserving bitwise; covered by the bundled re-anchor).
        - Skipped trajs keep mu/sigma frozen; MPC zeroes their taken actions anyway.

        Known (accepted) semantic differences under skip_succeeded:
        - The logged {prefix}/loss averages over ACTIVE trajs only (the stock loop
          averages all n_evals) -- loss curves are not comparable across the two modes.
        - The inner-eval all-success early break may stop firing: eval_actions(mu)
          re-rolls skipped trajs' FROZEN mu from the current state, which can read as
          non-success even though MPC already locked them. Harmless (the skipped work
          is the point); pair with eval_every=999 as the fast config does.
        """
        active_mask = (
            ~np.asarray(self.success_mask, dtype=bool)
            if (self.skip_succeeded and self.success_mask is not None)
            else np.ones(n_evals, dtype=bool)
        )
        active = [t for t in range(n_evals) if active_mask[t]]
        chunk_size = max(1, self.traj_chunk)
        if len(active) < n_evals:
            print(f"[cem] skip_succeeded: planning {len(active)}/{n_evals} trajs "
                  f"(traj_chunk={chunk_size})")

        for i in range(self.opt_steps):
            # 1) candidate draws: all trajs, traj order (byte-identical RNG stream)
            cand = []
            for traj in range(n_evals):
                action = (
                    torch.randn(self.num_samples, self.horizon, self.action_dim).to(
                        self.device
                    )
                    * sigma[traj]
                    + mu[traj]
                )
                action[0] = mu[traj]  # optional: make the first one mu itself
                cand.append(action)

            # 2) chunked rollout + per-traj scoring/refit (active trajs only)
            losses = []
            with torch.no_grad():
                for c0 in range(0, len(active), chunk_size):
                    chunk = active[c0 : c0 + chunk_size]
                    cur_z_obs_0 = {
                        key: torch.cat(
                            [
                                repeat(arr[t].unsqueeze(0), "1 ... -> n ...",
                                       n=self.num_samples)
                                for t in chunk
                            ],
                            dim=0,
                        )
                        for key, arr in z_obs_0.items()
                    }
                    act_chunk = torch.cat([cand[t] for t in chunk], dim=0)
                    i_z_obses, i_zs = self.wm.rollout_from_zobs(
                        z_obs_0=cur_z_obs_0,
                        act=act_chunk,
                    )
                    for j, traj in enumerate(chunk):
                        sl = slice(j * self.num_samples, (j + 1) * self.num_samples)
                        z_pred_traj = {key: arr[sl] for key, arr in i_z_obses.items()}
                        cur_z_obs_g = {
                            key: repeat(arr[traj].unsqueeze(0), "1 ... -> n ...",
                                        n=self.num_samples)
                            for key, arr in z_obs_g.items()
                        }
                        vis_mask = (None if self.patch_mask is None
                                    else self.patch_mask[traj])
                        loss = self.objective_fn(z_pred_traj, cur_z_obs_g,
                                                 vis_mask=vis_mask)
                        topk_idx = torch.argsort(loss)[: self.topk]
                        topk_action = cand[traj][topk_idx]
                        losses.append(loss[topk_idx[0]].detach())
                        mu[traj] = topk_action.mean(dim=0)
                        sigma[traj] = topk_action.std(dim=0)

            # single host sync per opt step (vs one per traj in the stock loop)
            loss_mean = (torch.stack(losses).double().mean().item()
                         if losses else float("nan"))
            self.wandb_run.log(
                {f"{self.logging_prefix}/loss": loss_mean, "step": i + 1}
            )
            if self.evaluator is not None and i % self.eval_every == 0:
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

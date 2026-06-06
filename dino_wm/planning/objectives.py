import numpy as np
import torch
import torch.nn as nn


def create_objective_fn(alpha, base, mode="last"):
    """
    Loss calculated on the last pred frame.
    Args:
        alpha: int
        base: int. only used for objective_fn_all
    Returns:
        loss: tensor (B, )
    """
    metric = nn.MSELoss(reduction="none")

    def _masked_visual_mean(se, vis_mask):
        """se: (B, T, P, D) squared error. vis_mask: None or (P,) 1=keep/0=drop.

        With a mask, average only over KEPT patches (+ feature dim) so the loss
        scale is comparable to the unmasked case. This is the manipulator-masked
        energy: drop the pusher patches the planner can't know the goal pose of.
        """
        if vis_mask is None:
            return se.mean(dim=tuple(range(1, se.ndim)))
        m = vis_mask.to(se.device).to(se.dtype).view(1, 1, -1, 1)  # (1,1,P,1)
        denom = m.sum() * se.shape[-1] * se.shape[1] + 1e-8
        return (se * m).sum(dim=tuple(range(1, se.ndim))) / denom

    def objective_fn_last(z_obs_pred, z_obs_tgt, vis_mask=None):
        """
        Args:
            z_obs_pred: dict, {'visual': (B, T, *D_visual), 'proprio': (B, T, *D_proprio)}
            z_obs_tgt: dict, {'visual': (B, T, *D_visual), 'proprio': (B, T, *D_proprio)}
            vis_mask: optional (P,) patch mask; 0 drops a patch from the energy.
        Returns:
            loss: tensor (B, )
        """
        se = metric(z_obs_pred["visual"][:, -1:], z_obs_tgt["visual"])
        loss_visual = _masked_visual_mean(se, vis_mask)
        loss_proprio = metric(z_obs_pred["proprio"][:, -1:], z_obs_tgt["proprio"]).mean(
            dim=tuple(range(1, z_obs_pred["proprio"].ndim))
        )
        loss = loss_visual + alpha * loss_proprio
        return loss

    def objective_fn_all(z_obs_pred, z_obs_tgt, vis_mask=None):
        """
        Loss calculated on all pred frames.
        Args:
            z_obs_pred: dict, {'visual': (B, T, *D_visual), 'proprio': (B, T, *D_proprio)}
            z_obs_tgt: dict, {'visual': (B, T, *D_visual), 'proprio': (B, T, *D_proprio)}
        Returns:
            loss: tensor (B, )
        """
        coeffs = np.array(
            [base**i for i in range(z_obs_pred["visual"].shape[1])], dtype=np.float32
        )
        coeffs = torch.tensor(coeffs / np.sum(coeffs)).to(z_obs_pred["visual"].device)
        se_v = metric(z_obs_pred["visual"], z_obs_tgt["visual"])  # (B,T,P,D)
        if vis_mask is None:
            loss_visual = se_v.mean(dim=tuple(range(2, se_v.ndim)))
        else:
            m = vis_mask.to(se_v.device).to(se_v.dtype).view(1, 1, -1, 1)
            loss_visual = (se_v * m).sum(dim=(2, 3)) / (m.sum() * se_v.shape[-1] + 1e-8)
        loss_proprio = metric(z_obs_pred["proprio"], z_obs_tgt["proprio"]).mean(
            dim=tuple(range(2, z_obs_pred["proprio"].ndim))
        )
        loss_visual = (loss_visual * coeffs).mean(dim=1)
        loss_proprio = (loss_proprio * coeffs).mean(dim=1)
        loss = loss_visual + alpha * loss_proprio
        return loss

    if mode == "last":
        return objective_fn_last
    elif mode == "all":
        return objective_fn_all
    else:
        raise NotImplementedError


def create_qm_objective_fn(alpha, base, mode, qm_head, w_qm=1.0, w_l2=1.0,
                           per_step=False):
    """Energy = w_l2 * masked-L2  +  w_qm * d_theta  ( + alpha * proprio ).

    The quasimetric term d_theta(mask(z_T), mask(z_goal)) is the dense, asymmetric,
    long-range cost-to-go (the learned replacement for the proprio shaping); the
    masked-L2 term provides final-pose precision. The SAME vis_mask (union of the
    goal + real pusher patches) is applied to BOTH latents inside the head -- exactly
    as it was applied in training -- so train/plan masking is consistent.

    This wraps the stock `objective_fn_last` energy (so masked-L2 and the optional
    proprio term are byte-identical to the floor) and only ADDS the w_qm * d_theta
    term. The CEM search math, mask plumbing, and success criterion are untouched.

    Args:
        qm_head: a trained, eval()-mode QuasimetricHead on the planner's device.
        w_qm, w_l2: energy weights (configurable; tune the ratio on the box).
        per_step: if True, average d_theta over ALL predicted rollout frames
                  (potential-style dense shaping); else terminal frame only.
    """
    base_fn = create_objective_fn(alpha, base, mode)
    qm_device = next(qm_head.parameters()).device

    def _d(z_grid, z_goal_grid, keep):
        # z_grid, z_goal_grid: (N, P, D); keep: (P,) or (N,P) or None
        return qm_head(z_grid.to(qm_device), z_goal_grid.to(qm_device),
                       None if keep is None else keep.to(qm_device))

    def objective_fn_qm(z_obs_pred, z_obs_tgt, vis_mask=None):
        # stock masked-L2 (+ proprio) energy, weighted by w_l2 (proprio keeps alpha)
        l2 = base_fn(z_obs_pred, z_obs_tgt, vis_mask=vis_mask)  # (B,)
        zp = z_obs_pred["visual"]                               # (B,T,P,D)
        zg = z_obs_tgt["visual"][:, -1]                         # (B,P,D)
        B = zp.shape[0]
        with torch.no_grad():
            if per_step:
                T = zp.shape[1]
                zp_flat = zp.reshape(B * T, *zp.shape[2:])      # (B*T,P,D)
                zg_rep = zg.unsqueeze(1).expand(B, T, *zg.shape[1:]).reshape(B * T, *zg.shape[1:])
                keep = None if vis_mask is None else vis_mask.view(1, -1).expand(B * T, -1)
                d = _d(zp_flat, zg_rep, keep).reshape(B, T).mean(dim=1)
            else:
                d = _d(zp[:, -1], zg, vis_mask)                 # (B,)
        return w_l2 * l2 + w_qm * d.to(l2.device)

    return objective_fn_qm

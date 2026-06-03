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

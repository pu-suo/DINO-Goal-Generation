"""Pure V* quasimetric cost-to-go head over MASKED DINO-WM object latents.

This is NOT the bridge `g`, and NOT QRL's latent transition / Q / policy. It is a
single map

    d_theta(z_a, z_b) = qm_head( proj( f(mask(z_a)) ),  proj( f(mask(z_b)) ) )

where everything upstream (the frozen DINOv2 encoder + DINO-WM dynamics) is
untouched. After QRL training on cached pusht_noise transitions (local cost
r = -1 per MODEL step), `-d_theta(z, z_goal)` approximates the goal-conditioned
optimal value V*(z; z_goal) in MODEL-STEP units, matching the planner's goal_H.

Design choices that matter (see docs/QUASIMETRIC_RUNBOOK.md for the full rationale):

* POSE-PRESERVING spatial encoder `f`. We do NOT global-mean-pool the 14x14 patch
  grid -- that destroys the T's position/orientation, which is the known Push-T
  bottleneck. `f` is a small conv stack over the (C, 14, 14) latent grid that keeps
  coarse spatial structure (down to 4x4) before flattening.

* MASKING IS APPLIED IDENTICALLY TO BOTH INPUTS. For any pair we drop the UNION of
  the two pushers' patches (the existing manipulator-masked-energy mask) from BOTH
  latents, so `d_theta` always sees two grids carrying the same keep-mask -- in
  training AND at planning time. This removes the train/plan mask mismatch the
  per-frame mask would otherwise introduce.

* SINGLE FRAME. `d_theta` consumes one latent state (B, P, D); it never takes the
  num_hist history axis. The dynamics model needs history to predict; the value
  head does not. Callers pass the *last* predicted frame and the goal frame.

The IQE quasimetric head is vendored at `third_party.torchqmet` (Wang & Isola
2022, IQE-maxmean). A symmetric-L2 head (same f + projector, Euclidean distance
instead of IQE) is provided for the asymmetric-vs-symmetric ablation.
"""

from typing import Optional, Union

import torch
import torch.nn as nn
from einops import rearrange

# vendored quasimetric embeddings (BSD-3, see third_party/torchqmet/LICENSE)
from third_party.torchqmet import IQE, MRN, MRNFixed


def apply_keep_mask(z: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """Zero the dropped patches of a latent grid.

    Args:
        z:    (..., P, D) latent grid.
        keep: (..., P) or (P,) float/bool, 1 = keep, 0 = drop. Broadcast over D.
    Returns:
        (..., P, D) with dropped patches set to 0.
    """
    keep = keep.to(z.dtype)
    if keep.dim() == 1:                       # (P,) -> broadcast over batch dims
        keep = keep.view(*([1] * (z.dim() - 2)), keep.shape[0])
    return z * keep.unsqueeze(-1)


class SpatialEncoder(nn.Module):
    """Pose-preserving conv encoder over a (P=grid^2, in_dim) DINO latent grid.

    Per-patch LayerNorm over the feature dim, optional appended keep-mask channel,
    then a small strided conv stack 14x14 -> 7x7 -> 4x4 (GroupNorm + GELU),
    flattened and projected to ``out_dim``. No global pooling.
    """

    def __init__(self, in_dim=384, grid=14, out_dim=256,
                 append_mask_channel=True, width=192, mid=256):
        super().__init__()
        self.in_dim = in_dim
        self.grid = grid
        self.append_mask_channel = append_mask_channel
        self.in_norm = nn.LayerNorm(in_dim)
        cin = in_dim + (1 if append_mask_channel else 0)
        ng = 16
        self.conv = nn.Sequential(
            nn.Conv2d(cin, width, 3, stride=1, padding=1),     # 14x14
            nn.GroupNorm(ng, width), nn.GELU(),
            nn.Conv2d(width, width, 3, stride=2, padding=1),   # 14 -> 7
            nn.GroupNorm(ng, width), nn.GELU(),
            nn.Conv2d(width, mid, 3, stride=2, padding=1),     # 7 -> 4
            nn.GroupNorm(ng, mid), nn.GELU(),
        )
        side = (grid + 1) // 2          # 14 -> 7
        side = (side + 1) // 2          # 7  -> 4
        self.head = nn.Linear(mid * side * side, out_dim)
        self.out_dim = out_dim

    def forward(self, z: torch.Tensor, keep: Optional[torch.Tensor] = None) -> torch.Tensor:
        """z: (B, P, in_dim). keep: (B, P) or (P,) 1=keep/0=drop (already applied
        to z by the caller; passed here only to build the appended mask channel)."""
        B, P, _ = z.shape
        x = self.in_norm(z)                                    # (B, P, in_dim)
        if self.append_mask_channel:
            if keep is None:
                ones = torch.ones(B, P, 1, device=z.device, dtype=x.dtype)
                x = torch.cat([x, ones], dim=-1)
            else:
                k = keep.to(x.dtype)
                if k.dim() == 1:
                    k = k.view(1, P).expand(B, P)
                x = torch.cat([x, k.unsqueeze(-1)], dim=-1)
        x = rearrange(x, "b (h w) c -> b c h w", h=self.grid, w=self.grid)
        x = self.conv(x)
        x = rearrange(x, "b c h w -> b (c h w)")
        return self.head(x)


class Projector(nn.Module):
    """MLP projector f_out -> hidden -> proj_out (the quasimetric input space)."""

    def __init__(self, in_dim=256, hidden=512, out_dim=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class QuasimetricHead(nn.Module):
    """d_theta(z_a, z_b) over masked object-only DINO latents. Pure V*; no T/Q/pi.

    head_type:
      - "iqe"      : torchqmet IQE-maxmean (asymmetric quasimetric). DEFAULT.
      - "mrn"      : torchqmet MRN (faster ~2x; ablation/fallback).
      - "mrn_fixed": torchqmet MRNFixed(sym_p=1) (guaranteed quasimetric).
      - "sym_l2"   : symmetric Euclidean distance in the SAME embedding space
                     (same f + projector). The asymmetry ablation: if this matches
                     "iqe", asymmetry is not buying anything.
    """

    def __init__(
        self,
        in_dim: int = 384,
        grid: int = 14,
        f_out: int = 256,
        proj_hidden: int = 512,
        proj_out: int = 2048,
        dim_per_component: int = 32,
        head_type: str = "iqe",
        append_mask_channel: bool = True,
        enc_width: int = 192,
        enc_mid: int = 256,
    ):
        super().__init__()
        assert head_type in ("iqe", "mrn", "mrn_fixed", "sym_l2"), head_type
        if head_type == "iqe":
            assert proj_out % dim_per_component == 0, (
                f"proj_out={proj_out} must be divisible by dim_per_component="
                f"{dim_per_component} (torchqmet IQE requirement)."
            )
        self.head_type = head_type
        self.grid = grid
        self.in_dim = in_dim
        self.append_mask_channel = append_mask_channel
        self.encoder = SpatialEncoder(
            in_dim=in_dim, grid=grid, out_dim=f_out,
            append_mask_channel=append_mask_channel, width=enc_width, mid=enc_mid,
        )
        self.projector = Projector(in_dim=f_out, hidden=proj_hidden, out_dim=proj_out)
        if head_type == "iqe":
            self.qm = IQE(proj_out, dim_per_component=dim_per_component)
        elif head_type == "mrn":
            # MRN has its own internal f_sym/f_asym projection over `input_size`.
            self.qm = MRN(proj_out)
        elif head_type == "mrn_fixed":
            self.qm = MRNFixed(proj_out)  # sym_p=1 -> guaranteed quasimetric
        else:
            self.qm = None  # sym_l2 computes Euclidean directly

    # -- embedding ------------------------------------------------------------
    def embed(self, z: torch.Tensor, keep: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Masked latent grid (B, P, D) -> embedding (B, proj_out).

        The caller is responsible for `keep` being the SAME mask used for the
        partner latent in any pairwise distance (union-of-pushers, see module
        docstring). Masking is applied here so callers pass the raw grid + keep.
        """
        if keep is not None:
            z = apply_keep_mask(z, keep)
        f = self.encoder(z, keep)
        return self.projector(f)

    # -- distance -------------------------------------------------------------
    def distance_from_embeddings(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if self.head_type == "sym_l2":
            return torch.linalg.vector_norm(u - v, dim=-1)
        return self.qm(u, v)

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor,
                keep: Optional[torch.Tensor] = None) -> torch.Tensor:
        """d_theta(z_a, z_b). z_a, z_b: (B, P, D). keep: (B,P) or (P,), applied to
        BOTH inputs identically. Returns (B,)."""
        u = self.embed(z_a, keep)
        v = self.embed(z_b, keep)
        return self.distance_from_embeddings(u, v)

    def distance(self, z_a, z_b, keep=None):
        return self.forward(z_a, z_b, keep)


def load_quasimetric_head(ckpt_path, device="cpu"):
    """Load a trained head saved by scripts/train_quasimetric.py.

    Returns (head in eval() on `device`, ckpt dict). The ckpt carries `head_cfg`
    (so the architecture is reconstructed exactly) and `mask_dilation` (so the
    planner can assert the energy mask matches what the head was trained with).
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    head = build_quasimetric_head(ckpt["head_cfg"]).to(device)
    head.load_state_dict(ckpt["state_dict"])
    head.eval()
    for p in head.parameters():
        p.requires_grad_(False)
    return head, ckpt


def build_quasimetric_head(cfg: Union[dict, object]) -> QuasimetricHead:
    """Construct a QuasimetricHead from a dict / OmegaConf-like config of kwargs."""
    get = (lambda k, d: cfg.get(k, d)) if isinstance(cfg, dict) else (lambda k, d: getattr(cfg, k, d))
    return QuasimetricHead(
        in_dim=get("in_dim", 384),
        grid=get("grid", 14),
        f_out=get("f_out", 256),
        proj_hidden=get("proj_hidden", 512),
        proj_out=get("proj_out", 2048),
        dim_per_component=get("dim_per_component", 32),
        head_type=get("head_type", "iqe"),
        append_mask_channel=get("append_mask_channel", True),
        enc_width=get("enc_width", 192),
        enc_mid=get("enc_mid", 256),
    )

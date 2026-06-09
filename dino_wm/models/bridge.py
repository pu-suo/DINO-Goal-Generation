"""`g` (the bridge): (z_start, text) -> z_goal, single forward pass, everything else frozen.

Definitive spec: specs/G_ARCHITECTURE.md (read it in full). This implements §3 (the
block-by-block bidirectional DiT forward), §4 (the weighted-L2 loss + changed-region mask),
and §6 (the frozen text encoder / trainable text_proj boundary).

CONTRACT (do NOT break):
- Input  z_start: (B, 196, 384) = DINOv2 x_norm_patchtokens (CLS stripped, LayerNorm applied,
  single frame -- NO num_hist time axis). Output z_goal lives in the SAME space.
- Output is the FULL grid via residual: z_goal = z_start + gate * Delta  (never object-only).
- Frozen: encoder, dynamics, CEM, text encoder. Trainable: text_proj, blocks, head, gate, pos_embed.
- No actions, no time axis, no autoregression, no causal mask. `g` is NOT the AC predictor.
"""
import math

import torch
import torch.nn as nn

N_PATCHES = 196
DIM = 384


# ----------------------------------------------------------------------------- block
class BridgeBlock(nn.Module):
    """One DiT-style block: bidirectional self-attn over patches, then cross-attn to text,
    then MLP. Pre-LN residual throughout (§3)."""

    def __init__(self, dim=DIM, heads=6, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.ln_sa = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ln_ca = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ln_mlp = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, dim), nn.Dropout(dropout))

    def forward(self, x, text_kv, text_key_padding_mask=None):
        # bidirectional self-attention over the 196 patch tokens (no mask)
        h = self.ln_sa(x)
        x = x + self.self_attn(h, h, h, need_weights=False)[0]
        # cross-attention: patches (query) attend to text tokens (key/value)
        q = self.ln_ca(x)
        x = x + self.cross_attn(q, text_kv, text_kv,
                                key_padding_mask=text_key_padding_mask, need_weights=False)[0]
        x = x + self.mlp(self.ln_mlp(x))
        return x


# ------------------------------------------------------------------------------ g
class BridgeG(nn.Module):
    """The bridge `g`. forward(z_start, text_tokens, text_mask) -> z_goal.

    `text_tokens` is the FROZEN text encoder's token-level output (B, L, d_text); `text_mask`
    is (B, L) bool with True = real token (False = pad). The frozen text encoder lives OUTSIDE
    this module (see FrozenTextEncoder); g owns only the trainable text_proj + transformer.
    """

    def __init__(self, dim=DIM, depth=6, heads=6, mlp_ratio=4, d_text=DIM,
                 n_patches=N_PATCHES, dropout=0.0, residual=True, gate_per_patch=True):
        super().__init__()
        self.dim = dim
        self.n_patches = n_patches
        self.residual = residual

        # learned patch positional embedding (§3: learned [1,196,384])
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # trainable text projection d_text -> dim (§2/§6: small MLP)
        self.text_proj = nn.Sequential(nn.Linear(d_text, dim), nn.GELU(), nn.Linear(dim, dim))

        self.blocks = nn.ModuleList([
            BridgeBlock(dim, heads, mlp_ratio, dropout) for _ in range(depth)])

        self.head_norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, dim)

        # zero-init residual gate (§3/§7): at init Delta is gated to 0 -> z_goal == z_start.
        gate_shape = (1, n_patches, 1) if gate_per_patch else (1, 1, 1)
        self.gate = nn.Parameter(torch.zeros(gate_shape))

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, z_start, text_tokens, text_mask=None):
        """z_start: (B,196,384); text_tokens: (B,L,d_text); text_mask: (B,L) bool True=real."""
        assert z_start.shape[-2:] == (self.n_patches, self.dim), \
            f"z_start must be (B,{self.n_patches},{self.dim}), got {tuple(z_start.shape)}"
        x = z_start + self.pos_embed
        text_kv = self.text_proj(text_tokens)
        # nn.MultiheadAttention key_padding_mask: True = IGNORE -> invert the real-token mask
        kpm = (~text_mask) if text_mask is not None else None
        for block in self.blocks:
            x = block(x, text_kv, text_key_padding_mask=kpm)
        raw_delta = self.head(self.head_norm(x))
        delta = self.gate * raw_delta
        return (z_start + delta) if self.residual else delta


# --------------------------------------------------------------------------- loss
def changed_region_mask(z_start, z_target, tau):
    """(B,196) float mask of patches that moved between start and goal (§4/§5).

    d_i = ||t_i - s_i||_2 over the 384 feature dim; changed_i = 1[d_i > tau].
    Used ONLY in g's training loss (up-weighting) -- never in the planning energy.
    """
    d = (z_target - z_start).norm(dim=-1)            # (B,196)
    return (d > tau).float()


def estimate_tau(z_start, z_target, bins=256):
    """Otsu threshold on the pooled per-patch change magnitudes d_i (§4).

    Splits the bimodal d_i distribution (background ~0  vs  moved-T/origin patches) at the
    valley. Compute ONCE on the dataset (or per-batch); not learned. Log the histogram early.
    """
    d = (z_target - z_start).norm(dim=-1).flatten()
    d = d[torch.isfinite(d)]
    lo = float(d.min())
    hi = float(d.quantile(0.999))
    if hi <= lo:
        return lo
    hist = torch.histc(d.clamp(lo, hi), bins=bins, min=lo, max=hi)
    p = hist / hist.sum().clamp_min(1e-12)
    edges = torch.linspace(lo, hi, bins)
    omega = torch.cumsum(p, 0)                        # class-0 prob mass up to bin k
    mu = torch.cumsum(p * edges, 0)
    mu_t = mu[-1]
    denom = (omega * (1.0 - omega)).clamp_min(1e-12)
    sigma_b2 = (mu_t * omega - mu) ** 2 / denom       # between-class variance
    k = int(torch.argmax(torch.nan_to_num(sigma_b2)))
    return float(edges[k])


def bridge_loss(z_goal_pred, z_target, z_start, tau, lam=7.0, reduction="mean"):
    """Weighted-mean L2 to enc(o_goal), up-weighted on the changed patches (§4).

        weight_i = 1 + lam * 1[ ||t_i - s_i|| > tau ]
        L = mean_b ( sum_i weight_i * ||zhat_i - t_i||^2 / sum_i weight_i )

    The WHOLE grid is supervised (static patches weight 1 -> Delta held near 0 there); the
    up-weight focuses capacity on the moved-T + origin erasure. L2 is the DINO-WM native metric.
    """
    changed = changed_region_mask(z_start, z_target, tau)        # (B,196)
    weight = 1.0 + lam * changed                                 # (B,196)
    per_patch_sq = (z_goal_pred - z_target).pow(2).sum(-1)       # (B,196)
    per_sample = (weight * per_patch_sq).sum(-1) / weight.sum(-1).clamp_min(1e-12)  # (B,)
    if reduction == "mean":
        return per_sample.mean()
    if reduction == "none":
        return per_sample
    raise ValueError(reduction)


# ------------------------------------------------------------------- frozen text enc
class FrozenTextEncoder(nn.Module):
    """Frozen sentence-transformer (default all-MiniLM-L6-v2, d_text=384). Token-level outputs.

    §6: frozen (eval, requires_grad=False), loaded once; grounding happens in g's cross-attention,
    NOT by fine-tuning this. Lazy-imports transformers so the rest of bridge.py has no hard dep.
    """

    def __init__(self, name="sentence-transformers/all-MiniLM-L6-v2", max_len=16, device="cpu"):
        super().__init__()
        from transformers import AutoTokenizer, AutoModel
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).eval().to(device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.max_len = max_len
        self.d_text = self.model.config.hidden_size

    @torch.no_grad()
    def forward(self, texts):
        """list[str] (len B) -> (text_tokens (B,L,d_text), text_mask (B,L) bool True=real)."""
        enc = self.tokenizer(list(texts), padding="max_length", truncation=True,
                             max_length=self.max_len, return_tensors="pt")
        dev = next(self.model.parameters()).device
        enc = {k: v.to(dev) for k, v in enc.items()}
        tokens = self.model(**enc).last_hidden_state           # (B, L, d_text)
        return tokens, enc["attention_mask"].bool()

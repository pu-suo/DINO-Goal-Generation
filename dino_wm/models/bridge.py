"""`g` (the bridge): (z_start, spec) -> z_goal, single forward pass, everything else frozen.

Definitive spec: specs/G_ARCHITECTURE.md + the clean-scene COORDINATE pivot (Option A,
docs/CLEAN_SCENE_PIVOT.md). This implements §3 (the block-by-block bidirectional DiT
forward), §4 (the weighted-L2 loss + changed-region mask), §6 (the frozen text encoder),
AND the coordinate front-end (CoordSpecEncoder): Fourier (x,y) + (sin,cos) theta tokens
cross-attended at every block, plus a soft 2D Gaussian heatmap over the 14x14 patch grid
added as a spatial positional bias. ONE module, two conditioning front-ends (cond_mode):
'coord' (primary) and 'text' (secondary); everything after cross-attn (self-attn stack,
per-patch zero-init gate, residual head, loss) is SHARED verbatim.

CONTRACT (do NOT break):
- Input  z_start: (B, 196, 384) = DINOv2 x_norm_patchtokens (CLS stripped, single frame --
  NO num_hist time axis). Output z_goal lives in the SAME space.
- Output is the FULL grid via residual: z_goal = z_start + gate * Delta  (never object-only).
- Frozen: encoder, dynamics, CEM, text encoder. Trainable: spec/text front-end, blocks, head, gate, pos_embed.
- No actions, no time axis, no autoregression, no causal mask. `g` is NOT the AC predictor.
"""
import math

import torch
import torch.nn as nn

N_PATCHES = 196
DIM = 384
GRID = 14         # 14x14 = 196 patches, row-major token = ri*GRID + ci (ci=col<->x, ri=row<->y)
SIM = 512.0       # PushT sim-space side length


def sim_xy_to_grid(xy_sim, grid=GRID, sim=SIM):
    """Map sim-512 (x,y) -> continuous patch-grid coords (col, row), matching
    env.pusht.multicolor_common.pusher_patch_mask EXACTLY (col<->x, row<->y, linear
    scale, no flip; token index = row*grid + col). xy_sim: (...,2) -> (col,row) (...,2)."""
    return xy_sim / sim * grid


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


# ----------------------------------------------------------------- coord front-end
class CoordSpecEncoder(nn.Module):
    """Coordinate spec front-end (clean-scene pivot). Turns a goal pose (x,y,theta)
    [+ optional extent] in sim-512 coords into (a) a small set of cross-attention
    K/V tokens and (b) a soft 2D Gaussian heatmap over the 14x14 patch grid used as a
    spatial positional bias on the patch tokens.

    - (x,y) -> NeRF-style Fourier features (normalized to [0,1]) -> a POSITION token.
    - theta -> (sin,cos) (removes the 0/2pi wraparound) -> an ORIENTATION token.
    - extent -> a scalar token (0 in the point regime; extent only widens the scoring
      tolerance at eval, never the synthesis geometry -- so it stays ~0 here).
    - heatmap: Gaussian centered at the goal patch (col,row) -> per-patch additive bias
      (zero heatmap -> zero bias; init keeps the residual identity since the gate is 0).
    """

    def __init__(self, dim=DIM, n_freq=12, grid=GRID, sim=SIM, sigma=1.2):
        super().__init__()
        self.dim, self.grid, self.sim, self.sigma = dim, grid, sim, sigma
        self.register_buffer("freqs", (2.0 ** torch.arange(n_freq)) * math.pi, persistent=False)
        pos_in = 2 * 2 * n_freq                       # x,y each -> [sin,cos] over n_freq
        self.pos_mlp = nn.Sequential(nn.Linear(pos_in, dim), nn.GELU(), nn.Linear(dim, dim))
        self.ori_mlp = nn.Sequential(nn.Linear(2, dim), nn.GELU(), nn.Linear(dim, dim))
        self.ext_mlp = nn.Sequential(nn.Linear(1, dim), nn.GELU(), nn.Linear(dim, dim))
        self.type_emb = nn.Parameter(torch.zeros(3, dim))     # pos / ori / ext token-type
        nn.init.trunc_normal_(self.type_emb, std=0.02)
        self.heat_vec = nn.Linear(1, dim, bias=False)         # per-patch heatmap -> bias vector
        # patch centers in (col,row) units; token t -> ci=t%grid (col<->x), ri=t//grid (row<->y)
        idx = torch.arange(grid * grid)
        centers = torch.stack([(idx % grid) + 0.5, (idx // grid) + 0.5], dim=-1)  # (196,2)=(col,row)
        self.register_buffer("centers", centers, persistent=False)

    def _fourier(self, v):                             # v in [0,1], (B,) -> (B, 2*n_freq)
        a = v.unsqueeze(-1) * self.freqs
        return torch.cat([a.sin(), a.cos()], dim=-1)

    def tokens(self, spec):                            # spec (B,>=3) sim-512 -> (B,3,dim)
        x = (spec[:, 0] / self.sim).clamp(0, 1)
        y = (spec[:, 1] / self.sim).clamp(0, 1)
        th = spec[:, 2]
        ext = spec[:, 3] if spec.shape[1] > 3 else torch.zeros_like(x)
        pos_t = self.pos_mlp(torch.cat([self._fourier(x), self._fourier(y)], dim=-1)) + self.type_emb[0]
        ori_t = self.ori_mlp(torch.stack([th.sin(), th.cos()], dim=-1)) + self.type_emb[1]
        ext_t = self.ext_mlp(ext.unsqueeze(-1)) + self.type_emb[2]
        return torch.stack([pos_t, ori_t, ext_t], dim=1)

    def heatmap_bias(self, spec):                      # spec (B,>=3) -> (B,196,dim)
        tgt = sim_xy_to_grid(spec[:, :2], self.grid, self.sim).unsqueeze(1)   # (B,1,2) (col,row)
        d2 = ((self.centers.unsqueeze(0) - tgt) ** 2).sum(-1)                 # (B,196)
        h = torch.exp(-d2 / (2.0 * self.sigma ** 2))                          # (B,196)
        return self.heat_vec(h.unsqueeze(-1))                                 # (B,196,dim)


# ------------------------------------------------------------------------------ g
class BridgeG(nn.Module):
    """The bridge `g`. forward(z_start, text_tokens, text_mask) -> z_goal.

    `text_tokens` is the FROZEN text encoder's token-level output (B, L, d_text); `text_mask`
    is (B, L) bool with True = real token (False = pad). The frozen text encoder lives OUTSIDE
    this module (see FrozenTextEncoder); g owns only the trainable text_proj + transformer.
    """

    def __init__(self, dim=DIM, depth=6, heads=6, mlp_ratio=4, d_text=DIM,
                 n_patches=N_PATCHES, dropout=0.0, residual=True, gate_per_patch=True,
                 cond_mode="text", n_freq=12, heat_sigma=1.2):
        super().__init__()
        self.dim = dim
        self.n_patches = n_patches
        self.residual = residual
        self.cond_mode = cond_mode

        # learned patch positional embedding (§3: learned [1,196,384])
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # conditioning front-end: 'coord' (primary, CoordSpecEncoder) or 'text' (MiniLM proj)
        if cond_mode == "coord":
            self.spec_enc = CoordSpecEncoder(dim, n_freq=n_freq, sigma=heat_sigma)
        elif cond_mode == "text":
            self.text_proj = nn.Sequential(nn.Linear(d_text, dim), nn.GELU(), nn.Linear(dim, dim))
        else:
            raise ValueError(f"cond_mode must be 'coord' or 'text', got {cond_mode}")

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

    def _trunk(self, z_start, kv, kpm=None, spatial_bias=None):
        """Shared core: patch self-attn stack cross-attending to `kv`, residual head with
        per-patch zero-init gate. `spatial_bias` (B,196,dim) is added to the patch tokens
        (coord heatmap). The zero-init gate makes z_goal == z_start at init regardless of kv/bias."""
        assert z_start.shape[-2:] == (self.n_patches, self.dim), \
            f"z_start must be (B,{self.n_patches},{self.dim}), got {tuple(z_start.shape)}"
        x = z_start + self.pos_embed
        if spatial_bias is not None:
            x = x + spatial_bias
        for block in self.blocks:
            x = block(x, kv, text_key_padding_mask=kpm)
        raw_delta = self.head(self.head_norm(x))
        delta = self.gate * raw_delta
        return (z_start + delta) if self.residual else delta

    def forward_coord(self, z_start, spec):
        """COORD path (primary). z_start: (B,196,384); spec: (B,>=3) = (x,y,theta[,extent]) sim-512."""
        assert self.cond_mode == "coord", "model built with cond_mode='text'; use forward()"
        kv = self.spec_enc.tokens(spec)              # (B,3,dim) cross-attn K/V
        bias = self.spec_enc.heatmap_bias(spec)      # (B,196,dim) spatial positional bias
        return self._trunk(z_start, kv, kpm=None, spatial_bias=bias)

    def forward(self, z_start, text_tokens, text_mask=None):
        """TEXT path (secondary). z_start: (B,196,384); text_tokens: (B,L,d_text); text_mask (B,L) bool True=real."""
        assert self.cond_mode == "text", "model built with cond_mode='coord'; use forward_coord()"
        kv = self.text_proj(text_tokens)
        # nn.MultiheadAttention key_padding_mask: True = IGNORE -> invert the real-token mask
        kpm = (~text_mask) if text_mask is not None else None
        return self._trunk(z_start, kv, kpm=kpm)


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

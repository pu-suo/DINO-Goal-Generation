# adapted from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange, repeat

# helpers
NUM_FRAMES = 1
NUM_PATCHES = 1

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def generate_mask_matrix(npatch, nwindow):
    zeros = torch.zeros(npatch, npatch)
    ones = torch.ones(npatch, npatch)
    rows = []
    for i in range(nwindow):
        row = torch.cat([ones] * (i+1) + [zeros] * (nwindow - i-1), dim=1)
        rows.append(row)
    mask = torch.cat(rows, dim=0).unsqueeze(0).unsqueeze(0)
    return mask

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        # device guard: identical on CUDA boxes; allows CPU-only construction (local tests)
        self.bias = generate_mask_matrix(NUM_PATCHES, NUM_FRAMES).to(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        # Opt-in fused-attention fast path (docs/PLANNING_SPEED_PROFILE.md, fast config).
        # Read via getattr in forward: instances UNPICKLED from a checkpoint never run
        # this __init__, so the attribute may be absent -> default False (stock path).
        self.use_sdpa = False

    def forward(self, x):
        (
            B,
            T,
            C,
        ) = x.size()
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        # SDPA fast path: same math (default softmax scale = dim_head**-0.5 = self.scale)
        # without materializing the (B, h, T, T) score tensor. Guard: NEVER take it while
        # dropout would be active (train mode, p>0) -- F.sdpa's fused dropout draws from a
        # different RNG scheme than nn.Dropout, which would silently change results.
        # The mask is ADDITIVE float (0 / -inf) in q.dtype rather than bool: torch 2.3's
        # memory-efficient backend accepts float attn_bias broadly, while bool masks risk
        # a silent math-backend fallback that re-materializes the score tensor. The (T,T)
        # mask build per forward is negligible (~346 KB at T=588).
        if getattr(self, "use_sdpa", False) and not (self.training and self.dropout.p > 0):
            bias = self.bias[:, :, :T, :T]
            attn_mask = torch.zeros_like(bias, dtype=q.dtype).masked_fill(
                bias == 0, float("-inf")
            )
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        else:
            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            # apply causal mask
            dots = dots.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))

            attn = self.attend(dots)
            attn = self.dropout(attn)

            out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

def enable_sdpa(module, enabled=True):
    """Flip the SDPA fast path on every Attention block under `module`.

    Works on modules UNPICKLED from a checkpoint (sets the instance attribute that
    forward() reads via getattr). Returns the number of Attention blocks touched, so
    callers can verify the flag actually reached a ViT predictor (0 = nothing to do).
    """
    n = 0
    for m in module.modules():
        if isinstance(m, Attention):
            m.use_sdpa = bool(enabled)
            n += 1
    return n


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)
    
class ViTPredictor(nn.Module):
    def __init__(self, *, num_patches, num_frames, dim, depth, heads, mlp_dim, pool='cls', dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        
        # update params for adding causal attention masks
        global NUM_FRAMES, NUM_PATCHES
        NUM_FRAMES = num_frames
        NUM_PATCHES = num_patches

        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames * (num_patches), dim)) # dim for the pos encodings
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool

    def forward(self, x): # x: (b, window_size * H/patch_size * W/patch_size, 384)
        b, n, _ = x.shape
        x = x + self.pos_embedding[:, :n]
        x = self.dropout(x) 
        x = self.transformer(x) 
        return x
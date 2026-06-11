"""forward() vs forward_latent() numerical equivalence (CPU, stub encoders).

The cached-latent training path (forward_latent) must produce the SAME loss and z_pred
as the image path (forward) when fed the encoder's own output. Stub encoders make this
fast and deterministic; the cache-equals-encode_obs half is verified on-box with a
cosine selfcheck (scripts/cache_dynamics_latents.py --selfcheck).
"""
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.visual_world_model import VWorldModel

B, NF, NH, NP_PRED = 2, 4, 3, 1
P, D = 196, 16          # patches, encoder emb dim (tiny)
PROP_RAW, ACT_RAW = 4, 10
PROP_EMB, ACT_EMB = 6, 8


class StubEncoder(nn.Module):
    """Deterministic (N,3,h,w) -> (N,P,D); mimics a frozen DINOv2 (name has 'dino')."""
    name, patch_size, emb_dim = "dino_stub", 14, D

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(3 * 14 * 14, P * D)
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        x = torch.nn.functional.adaptive_avg_pool2d(x, (14, 14)).reshape(x.shape[0], -1)
        return self.proj(x).reshape(x.shape[0], P, D)


class MLPEnc(nn.Module):
    def __init__(self, din, demb):
        super().__init__()
        self.net = nn.Linear(din, demb)
        self.emb_dim = demb

    def forward(self, x):
        return self.net(x)


class Predictor(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Linear(dim, dim)

    def forward(self, z):  # (b, t*p, dim)
        return self.net(z)


def build_model():
    dim = D + PROP_EMB + ACT_EMB  # concat_dim=1 token width
    m = VWorldModel(
        image_size=224, num_hist=NH, num_pred=NP_PRED,
        encoder=StubEncoder(), proprio_encoder=MLPEnc(PROP_RAW, PROP_EMB),
        action_encoder=MLPEnc(ACT_RAW, ACT_EMB), decoder=None, predictor=Predictor(dim),
        proprio_dim=PROP_EMB, action_dim=ACT_EMB, concat_dim=1,
        num_action_repeat=1, num_proprio_repeat=1,
    )
    m.eval()  # custom train() override returns None, so don't chain
    return m


def test_forward_latent_matches_forward():
    torch.manual_seed(0)
    m = build_model()
    obs = {"visual": torch.randn(B, NF, 3, 224, 224),
           "proprio": torch.randn(B, NF, PROP_RAW)}
    act = torch.randn(B, NF, ACT_RAW)

    with torch.no_grad():
        zp_a, _, _, loss_a, comp_a = m(obs, act)
        z_vis = m.encode_obs(obs)["visual"]
        zp_b, _, _, loss_b, comp_b = m.forward_latent(z_vis, obs["proprio"], act)

    assert torch.allclose(loss_a, loss_b, atol=1e-6), (loss_a.item(), loss_b.item())
    assert torch.allclose(zp_a, zp_b, atol=1e-6)
    for k in ("z_loss", "z_visual_loss", "z_proprio_loss"):
        assert torch.allclose(comp_a[k], comp_b[k], atol=1e-6), k
    print(f"OK: forward {loss_a.item():.6f} == forward_latent {loss_b.item():.6f}")


if __name__ == "__main__":
    test_forward_latent_matches_forward()
    print("ALL OK")

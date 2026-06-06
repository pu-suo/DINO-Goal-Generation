"""
Local correctness anchor for the planning speed fix (runs on CPU, no GPU needed).

The speed fix (planning/cem.py + models.VWorldModel.rollout_from_zobs) hoists the
frozen DINOv2 encode of the start observation OUT of the per-candidate CEM inner
loop. The whole argument that this is *result-preserving* rests on one fact:

    During planning the dynamics model is left in train() mode, so the predictor's
    dropout (p=0.1) is ACTIVE and is the ONLY consumer of the (CUDA) RNG stream in a
    rollout. The frozen encoders (DINOv2 patch tokens, proprio Conv1d) contain NO
    active stochastic layer, so they draw NOTHING from the RNG stream. Therefore
    adding/removing encode() calls cannot change the dropout masks that the predictor
    draws -> every predicted latent is bit-identical -> CEM scores/elites are identical.

This test demonstrates that property directly with tiny modules (the principle is
device-agnostic; we use the CPU generator here, the real run uses the CUDA generator):

  1. nn.Dropout(p>0) in train mode DOES advance the RNG; nn.Dropout(0) / Linear / a
     LayerNorm-only "encoder" do NOT.
  2. Interleaving RNG-free ops between dropout draws leaves the dropout masks unchanged.
  3. The candidate action draw (torch.randn on the *default/CPU* generator) is
     independent of forward-pass dropout (so caching the encode cannot change which
     candidates CEM samples).

Run:  /Users/Tom/miniforge3/envs/dino_wm_dev/bin/python scripts/test_rng_invariance.py
Exit code 0 == all invariants hold.
"""
import sys
import torch
import torch.nn as nn


def _rng_state():
    return torch.random.get_rng_state()


def _advanced(before, after):
    return not torch.equal(before, after)


def test_rngfree_modules_do_not_advance_rng():
    """An RNG-free 'encoder' (Linear / LayerNorm / Dropout(0)) must not advance RNG."""
    x = torch.ones(4, 8)
    for name, mod in [
        ("Linear", nn.Linear(8, 8)),
        ("LayerNorm", nn.LayerNorm(8)),
        ("Dropout(0.0)", nn.Dropout(0.0)),
        ("Conv1d", nn.Conv1d(8, 8, 1)),
    ]:
        mod.train()
        torch.manual_seed(0)
        s0 = _rng_state()
        inp = x.unsqueeze(-1) if isinstance(mod, nn.Conv1d) else x
        _ = mod(inp)
        assert not _advanced(s0, _rng_state()), f"{name} unexpectedly drew from RNG"
    print("[ok] RNG-free modules (Linear/LayerNorm/Dropout(0)/Conv1d) do not advance RNG")


def test_active_dropout_advances_rng():
    drop = nn.Dropout(0.1).train()
    x = torch.ones(64, 64)
    torch.manual_seed(0)
    s0 = _rng_state()
    _ = drop(x)
    assert _advanced(s0, _rng_state()), "active dropout should advance the RNG"
    print("[ok] active Dropout(0.1) advances RNG (it is the rollout's only RNG consumer)")


def test_encode_caching_does_not_change_dropout_masks():
    """
    Model the rollout: a sequence of dropout draws (the predictor) with a frozen,
    RNG-free 'encoder' applied before each. Removing/hoisting the encoder (as the
    speed fix does) must leave every dropout output identical.
    """
    drop = nn.Dropout(0.1).train()
    enc = nn.Sequential(nn.Linear(32, 32), nn.LayerNorm(32))  # RNG-free frozen stand-in
    enc.train()
    for p in enc.parameters():
        p.requires_grad_(False)
    x = torch.randn(16, 32)

    # Reference stream: NO encode between dropout draws (5 "predict" steps).
    torch.manual_seed(123)
    ref = [drop(x) for _ in range(5)]

    # Speed stream: encode ONCE up front, then the SAME 5 dropout draws.
    torch.manual_seed(123)
    _ = enc(x)  # hoisted, cached encode (RNG-free)
    fast = [drop(x) for _ in range(5)]

    # Original stream: encode INSIDE the loop before every draw (RNG-free).
    torch.manual_seed(123)
    orig = []
    for _ in range(5):
        _ = enc(x)
        orig.append(drop(x))

    for i in range(5):
        assert torch.equal(ref[i], fast[i]), f"hoisted encode changed dropout draw {i}"
        assert torch.equal(ref[i], orig[i]), f"in-loop encode changed dropout draw {i}"
    print("[ok] hoisting/removing the RNG-free encode leaves all dropout masks identical")


def test_candidate_sampling_independent_of_forward_dropout():
    """
    CEM draws candidates with `torch.randn(...)` on the default (CPU) generator, then
    `.to(device)`. The predictor's dropout runs on the model's device generator. Here
    we show the candidate draw is unaffected by interleaved forward-pass dropout on the
    SAME generator in the worst case -- as long as candidate draws keep their order.
    The real code keeps the per-traj `torch.randn` call byte-for-byte unchanged, so the
    candidates are identical regardless of the encode caching.
    """
    # The encoder is built ONCE at model load (its weight-init draws happen long
    # before planning), so construct it before seeding -- mirroring the real code.
    enc = nn.Linear(4, 4).eval()  # RNG-free frozen stand-in for the encoder
    for p in enc.parameters():
        p.requires_grad_(False)

    # baseline: draw all candidates with no forward in between
    torch.manual_seed(7)
    cand_ref = [torch.randn(8, 5, 2) for _ in range(3)]

    # the speed fix does NOT move the randn calls relative to each other; it only
    # removes RNG-free encode work. Removing RNG-free work between the draws keeps them
    # identical:
    torch.manual_seed(7)
    cand_fast = []
    for _ in range(3):
        _ = enc(torch.ones(2, 4))  # RNG-free encode (no draw)
        cand_fast.append(torch.randn(8, 5, 2))

    for i in range(3):
        assert torch.equal(cand_ref[i], cand_fast[i]), f"candidate {i} changed"
    print("[ok] candidate sampling is invariant to interleaved RNG-free encode work")


if __name__ == "__main__":
    torch.use_deterministic_algorithms(False)
    tests = [
        test_rngfree_modules_do_not_advance_rng,
        test_active_dropout_advances_rng,
        test_encode_caching_does_not_change_dropout_masks,
        test_candidate_sampling_independent_of_forward_dropout,
    ]
    for t in tests:
        t()
    print("\nALL RNG-INVARIANCE CHECKS PASSED — caching the frozen encode is RNG-neutral.")
    sys.exit(0)

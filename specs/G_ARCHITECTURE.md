# docs/G_ARCHITECTURE.md — Definitive Build Spec for `g` (the bridge)

> **Read this in full before writing any `g` code (Phase 1).** This is the source of truth for the implementation; `CLAUDE.md` only summarizes it. `g` is built on a *frozen* DINO-WM (`gaoyuezhou/dino_wm`). When a repo API is referenced as `TODO: confirm`, you MUST read the actual source and pin it down before coding — do not guess.

---

## 0. Recap (one paragraph)
`g` maps `(z_start, text) → z_goal`, a single forward pass, with everything else frozen. It replaces "show the world model a goal image" with "tell it a goal in words." Its output is a **full per-patch goal latent grid**, produced as **start-plus-learned-change** (`z_goal = z_start + Δ`), in exactly the same representation the frozen encoder produces. The frozen DINO-WM dynamics + CEM planner then consume `z_goal` unchanged. `g` does grounding (find the named color, place the T there in latent space); CEM does control.

---

## 1. The frozen seam — interfaces to LOCATE and to NOT change
Before writing `g`, read the repo and fill in this table with verified facts (function names, shapes, dtypes). Confirm each by running a real batch and asserting shapes.

| What | Where (confirm) | Must capture |
|---|---|---|
| **Encoder** `enc(o) → patch latent` | `models/` encoder wrapper and/or `preprocessor.py` | exact output shape, dtype, **whether CLS is included or stripped**, any **normalization** applied to the latent (LayerNorm? per-channel mean/std?), and whether a **time/history axis** (`num_hist`) is present |
| **`enc(o_goal)`** (supervision target) | same encoder fn | must be byte-for-byte the same representation as `z_start` |
| **Dynamics / transition ViT** | `models/` | input/output latent shape it expects. `g` **never calls this** during its own forward/training; it is downstream-only. Do not modify it. |
| **Planner cost / goal entry** | `planning/` (+ `plan.py`) | where the goal latent enters, the **shape** it expects, and where the **per-patch cost is summed**. The manipulator-masked energy is a small edit *here*, behind a config flag. Do not touch CEM search logic. |
| **Success metric** | `metrics/` or `env/` | reuse for eval against the **named** target |

**Hard contract:** `g` consumes `z_start` and emits `z_goal` in the encoder's native representation. Do **not** re-encode, re-normalize, transpose, or reshape outside this contract. If the encoder applies a LayerNorm/standardization to the latent, `z_start`, `Δ`, and the loss target must all live in that same normalized space.

**`num_hist` gotcha:** DINO-WM's pipeline carries multiple history frames for the *dynamics* model. `g` uses a **single frame** (T=1) — the start frame's latent only. Strip/avoid the history axis inside `g`. If `z_start` arrives as `[B, T, 196, 384]`, take `T = -1` (or `T = 0` per the repo's convention — confirm) and operate on `[B, 196, 384]`.

---

## 2. Tensor shapes & dtypes at every interface
Batch-first throughout. `P = 196` patches (14×14, row-major — **confirm ordering** in the encoder), `D = 384`.

| Tensor | Shape | Notes |
|---|---|---|
| `o_0`, `o_goal` (frames) | `[B, 3, 224, 224]` | confirm input resize/normalization pipeline |
| `z_start = enc(o_0)` | `[B, 196, 384]` f32 | confirm CLS/normalization/history axis (see §1) |
| `text` (raw) | `list[str]`, len `B` | templated + paraphrased (from Phase 0) |
| `text_tokens` (frozen text enc.) | `[B, L, d_text]` | token-level, NOT pooled. `L ≈ 16` max. e.g. MiniLM `d_text = 384` |
| `text_mask` | `[B, L]` bool | key-padding mask for cross-attention |
| `text_kv` (projected) | `[B, L, 384]` | trainable Linear/MLP `d_text → 384` |
| `Δ` (raw head output) | `[B, 196, 384]` | the change |
| `z_goal = z_start + Δ` | `[B, 196, 384]` f32 | g's output; same space as `z_start` |
| `z_goal_target = enc(o_goal)` | `[B, 196, 384]` | supervision target |
| `changed_mask` | `[B, 196]` | from `‖target − start‖`; used in **loss only** |
| `manipulator_mask` | `[B, 196]` bool | pusher patches; used in **energy only** |

---

## 3. Module forward pass (block-by-block)
Bidirectional DiT-style transformer. **No causal mask, no time axis, no actions.**

```
# Hyperparameters
depth = 6 ; heads = 6 ; width = 384 ; mlp_ratio = 4 ; dropout = 0.0

# Embed
x = z_start + pos_embed                 # pos_embed: learned [1,196,384] or reuse DINOv2's patch pos-emb
text_kv = text_proj(text_encoder(text)) # text_encoder FROZEN; text_proj trainable (d_text -> 384)

# N blocks
for block in range(depth):
    x = x + SelfAttn(LN(x))                                   # spatial self-attn over 196 patches (bidirectional)
    x = x + CrossAttn(LN(x), kv=text_kv, key_padding_mask=text_mask)  # patches (query) attend to text (k/v)
    x = x + MLP(LN(x))                                        # ratio 4, GELU
    # adaLN-Zero ONLY if conditioning on a global scalar; otherwise plain LN above

# Residual head with zero-init gate (recommended)
raw_delta = head(LN(x))                  # Linear width->384  -> [B,196,384]
delta     = gate * raw_delta             # gate: learned per-patch (or scalar) in [0,1], INIT ~0
z_goal    = z_start + delta
```

**Zero-init gate (recommended, include):** initialize `gate ≈ 0` so at step 0, `z_goal == z_start` (identity). This makes early training stable, keeps the output on-manifold from the start, and operationalizes the residual framing. Unit-test that `z_goal` equals `z_start` at initialization.

**Conditioning rationale:** cross-attention (not adaLN) is required because the instruction is free-form/selective text that must be attended over per spatial position; adaLN is for global scalar conditions only.

---

## 4. Loss (exact)
Let `s = z_start`, `t = enc(o_goal)`, `ẑ = z_goal`, all `[B, 196, 384]`.

**Changed-region mask** (per patch `i`):
```
d_i        = ‖ t_i − s_i ‖_2          # L2 over the 384 feature dim
changed_i  = 1[ d_i > τ ]             # τ from data: histogram d_i, set τ at the valley between
                                      #   background (~0) and the T/origin patches
weight_i   = 1 + λ * changed_i        # λ ∈ [5, 10]
```

**Weighted-mean L2 loss:**
```
per_patch_sq_i = ‖ ẑ_i − t_i ‖_2^2                       # sum over 384 features
L = mean_over_batch( Σ_i weight_i * per_patch_sq_i  /  Σ_i weight_i )
```
- **Metric:** L2 for DINO-WM (matches its native latent metric). Use **L1** only on V-JEPA-2-AC.
- **The whole grid is supervised**, not just changed patches: static patches carry weight 1, so `Δ` is held near 0 there (faithful copy); the up-weight just focuses capacity on the moved-T + origin-erasure. Because `z_goal = z_start + Δ`, static patches are easy to get right — this is the entropy-reduction mechanism in code.
- `τ` is computed once on the dataset (or per-batch from `t` and `s`), not learned. Log a histogram of `d_i` early to set it.

---

## 5. The two masks — DO NOT CONFUSE (highest-value section)
These serve different stages. Mixing them is the most likely silent bug.

| | **changed-region mask** | **manipulator mask** |
|---|---|---|
| Computed from | `‖enc(o_goal) − z_start‖ > τ` | env's known pusher pose → occupied patches (preferred), else heuristic/learned segmentation on `z_start` |
| Applied in | **`g`'s TRAINING LOSS** (up-weighting) | **CEM ENERGY only** (exclude those patches from the cost sum) |
| Purpose | focus learning on moved-T + origin erasure | don't penalize the arm's unknown goal-time position |
| Available at | train time (uses the real target) | plan time |

- The **manipulator mask is NEVER applied to `g`'s training loss.**
- The **changed-region mask is NEVER applied to the planning energy.**
- For Phase-1 PushT, derive the manipulator mask **geometrically from the env state** (the pusher pose is exposed). Heuristic color-threshold or a tiny learned mask head are fallbacks only.

---

## 6. Text side — frozen / trainable boundary
- **Frozen:** the text encoder (recommend `all-MiniLM-L6-v2`, `d_text=384`, or a frozen CLIP text encoder). `eval()`, `requires_grad=False`, loaded once.
- **Trainable:** only `text_proj` (`d_text → 384`) and the transformer blocks + head + gate + pos_embed.
- Tokenize with the encoder's own tokenizer; keep **token-level** outputs + attention mask (not the pooled vector).
- Instructions are templated + paraphrased (Phase 0). **Hold out some paraphrase templates at test** to check meaning-grounding vs surface-form memorization.
- **Do NOT fine-tune the text encoder.** Grounding happens inside `g`'s cross-attention.

---

## 7. Optional components — include / defer
| Component | Decision | Notes |
|---|---|---|
| Zero-init residual gate (§3) | **INCLUDE** | stable init, on-manifold start |
| On-manifold regularization | **INCLUDE (lightweight)** | residual framing already helps; add an explicit penalty toward the nearest real encoded latent **only if** Stage-1 fidelity shows drift. A stronger option that also helps the planner: train/augment the frozen dynamics' *inputs* with light latent noise so CEM tolerates `g`'s imperfections (see Stage-2 fallback) |
| Retrieval-to-nearest-real-latent | **INCLUDE (eval baseline)** | not part of `g`; at eval, also plan toward the nearest real goal latent. If it beats raw-`g` a lot → `g` is off-manifold |
| Learned distance / calibration head | **DEFER** | only for the broader-generalization claim; do NOT build in Phase 1 |
| Generative (flow-matching) head | **DEFER / SKIP for PushT** | PushT goal is unimodal given the instruction; required only before high-entropy scaling (V-JEPA-2-AC, richer language) |

Wire each as a config flag. Defaults: residual gate **on**, retrieval baseline **on at eval**, everything else **off**.

---

## 8. Validation staging — what "done" means
| Stage | Test | Gate |
|---|---|---|
| **0** (Phase 0 prereq) | oracle (real-goal-image) planning on held-out combos | **oracle SR ≥ 0.80** before any `g` work |
| **1 — fidelity probe** | train `g`; measure *latent* fidelity decoupled from planning (changed-region cosine vs `enc(o_goal)`; decode `z_goal` and eyeball) | **changed-region cosine ≥ 0.9** and decoded goal recognizable as "T on the named target." If below → add up-weighting / consider flow head **before** touching CEM |
| **2 — closed-loop** | plug `g` into CEM with the **manipulator-masked** energy; compare vs oracle + retrieval baseline | **`g`-SR ≥ 0.75 absolute and ≥ 0.85× oracle** on held-out combos. If brittle despite good Stage-1 fidelity → off-manifold/energy problem → latent-noise aug on the dynamics, or swap CEM for a goal-conditioned inverse model |
| **3 — generalization + ablation** | held-out color-location combos; ablate residual-vs-direct head, cross-attn-vs-adaLN, frozen-vs-trained text, paraphrase on/off, `λ`, manipulator-mask on/off | the ablations ARE the paper's scientific content; each needs its own train + planning eval |

---

## 9. Build order for Phase 1 (do in this sequence)
1. **Confirm the frozen seam** (§1 table) with shape asserts on a real batch from the Phase-0 dataset.
2. **Implement `g`** in `models/bridge.py` per §3. Unit-test shapes; assert `z_goal == z_start` at init (zero-init gate).
3. **Implement the loss + changed-mask** (§4). **Overfit a single tiny batch to ~0 loss** as a sanity check.
4. **Train** on the Phase-0 dataset; run the **Stage-1 fidelity probe**. Do not proceed past the gate.
5. **Wire into CEM** (manipulator-masked energy, §5) only after Stage-1 passes → **Stage 2**.
6. **Held-out + ablations** → Stage 3.

---

## 10. Anti-patterns (g-specific do-NOTs)
- ❌ Actions, time axis, autoregression, causal mask, or rollout inside `g`. (That's the AC predictor; `g` is not it.)
- ❌ Object-only output. Always the **full grid via residual** (`z_start + Δ`). Object-only causes a double-T / off-manifold goal.
- ❌ Applying the manipulator mask to the loss, or the changed-region mask to the energy.
- ❌ Unfreezing the encoder, dynamics model, CEM, or text encoder.
- ❌ Carrying the `num_hist` time axis into `g` (single frame only).
- ❌ Reshaping / renormalizing the latent outside the encoder's contract.
- ❌ Fine-tuning the text encoder to "help" grounding.

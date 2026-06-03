# docs/RESEARCH_CONTEXT.md — Prior Art, Evidence & Success Estimate

> **Researcher reference, not build context.** An agent does **not** need this to implement `g`. It captures *why* the design is what it is and how likely it is to work, for positioning, related-work, and go/no-go decisions. Condensed from the verified June-2026 landscape pass; the full report lives in the project chat history.

---

## 1. Novelty (the one-line claim)
As of mid-2026, **no published system synthesizes a text-conditioned per-patch goal latent of a *frozen* JEPA/DINO dynamics world model and feeds it to that model's existing energy planner.** Every real neighbor uses a different mechanism. The space is crowded, so positioning must be sharp, but the specific mechanism is unoccupied.

## 2. Verified neighbors — cite and contrast (none scoop Paper 1)
| Work | arXiv | Mechanism | Why it differs from `g` |
|---|---|---|---|
| **SWM** (Semantic World Models) | 2510.19818 | VLM answers VQA about future semantic state; plans in action space | text-*out*, no goal latent at all — the cleanest foil |
| **PiJEPA** (Policy-Guided WM Planning for Lang-Cond. Visual Nav.) | 2603.25981 | language-conditioned policy warm-starts MPPI over a JEPA WM | language enters via the *policy*, bridge is in *action* space, navigation domain; still needs a goal/action distribution |
| **LCVN** (Lang-Cond. World Modeling for Visual Nav.) | 2603.26741 | diffusion WM + latent actor-critic / autoregressive multimodal | different mechanism + domain; no goal-latent-into-frozen-dynamics |
| **H-WM** (Hierarchical WM for TAMP) | 2602.11291 | LLM logical WM → visual WM emits latent visual subgoals to guide **VLAs** | closest to the broader vision; conditions on logical states, guides VLAs (not a frozen-WM CEM energy), full TAMP hierarchy. Erodes Paper 2, not Paper 1 |

## 3. Borrowables (use these)
| Work | arXiv | Use |
|---|---|---|
| **RAE** (Diffusion Transformers w/ Representation Autoencoders) | 2510.11690 | proves high-fidelity DINOv2 patch-latent generation is possible (rFID 0.49) — the core feasibility evidence. Also the key warning: naive generation on these latents catastrophically fails until the consumer tolerates imperfect/off-manifold latents → **motivates the residual head + on-manifold care** |
| **Talk2DINO** / dino.txt | 2411.19331 / 2412.16334 | text→DINOv2-patch grounding via a lightweight learned map on frozen backbones works → **supports frozen text encoder + small trainable bridge** |
| **SuSIE** | 2310.10639 | language→image-subgoal then goal-reaching beats an oracle-goal policy → closest *pixel-space* analogue; validates "synthesize goal from language, let a frozen executor run" |
| **GRIF** | 2307.00117 | aligns language to the start→goal *change* → **direct support for the `z_start + Δ` residual framing** |
| **GenTron** | 2312.04557 | cross-attention >> adaLN for free-form text conditioning → **supports cross-attention in `g`** (adaLN is for class/scalar conditions, per DiT 2212.09748) |
| **Value-guided JEPA planning** | 2601.00844 | shapes the latent so distance ≈ value → the concrete instantiation of the deferred calibration head; adopt only for the broader claim |
| **GC-IDM** (Latent Geometry Beyond Search) | 2605.08732 | goal-conditioned inverse model replacing CEM; consumes exactly the goal latent `g` produces → **Stage-2 fallback if CEM is brittle** |
| **Closing the Train-Test Gap in World Models** | 2512.09929 | quantifies how off-distribution latents raise WM error (+18/+20/+30% gains from fixing it on PushT/PointMaze/Wall) → **motivates latent-noise augmentation as the Stage-2 robustness fix** |

## 4. Why semantic-space methods don't transfer here
The JEPA/DINO latent has no native language alignment (near-orthogonal to text). Methods that regress text into a *pooled, pre-aligned* CLIP/SigLIP space (e.g. LBP-style) are parasitic on that pre-alignment and hide cross-modal distortion inside a trainable policy. Here the consumer is **fixed arithmetic** (frozen predictor + L2), and the goal is a **structured per-patch grid**, not a pooled vector — so the grounding must be learned directly into the spatial DINO space.

**Important reframe (de-risks one worry):** by the time CEM computes its cost, both sides are DINO features — `g` is *trained to regress to* `enc(o_goal)` (real DINO features), and the predictor rolls out DINO features. The energy is **within-DINO**, not cross-modal. So the "cross-modal distance distortion" critique does **not** directly bite the energy; the cross-modal step is absorbed into `g`'s supervised target. The real risks are plain regression accuracy and on-manifold realism (below), which is why a calibration head is deferred, not central.

---

## 5. Calibrated success estimate
**Core criterion:** `g`-driven planning is competitive with oracle (goal-image) planning on **held-out color-location combinations** in multi-color PushT — concretely ≥ 0.75 absolute and ≥ 0.85× the oracle SR.

Five conditional sub-risks (probability each is **not** a fatal blocker):

| Sub-risk | P(not fatal) | Driver |
|---|---|---|
| Cross-modal regression fidelity (text → correct patch region) | ~0.85 | Talk2DINO/dino.txt; targets visible in-frame makes it selection, not hallucination |
| On-manifold realism of synthesized `z_goal` | ~0.75 | RAE shows it's achievable but needs care; residual framing helps |
| Energy/distance fidelity for CEM on a synthesized goal | ~0.70 | train-test-gap + value-guided JEPA show real degradation; CEM more robust than GD; fixable |
| Latent pose resolution at 14×14 (T **orientation**) | ~0.70 | coarse stride-14 grid; DINO-WM still hits ~0.90 PushT so position is fine; orientation marginal — **the least-quantified risk; the Phase-0 §0.3 probe targets it** |
| Generalization to held-out combos | ~0.80 | compositional selection over visible targets; paraphrase aug |

**Narrow PushT result: ~55–65%** (upper-half-weighted because the goal is unimodal given the instruction, all targets are in-frame, and the residual head keeps `z_goal` near-manifold by construction).

**Broader bridging approach** (V-JEPA-2-AC, real robots, richer/compositional language): **~35–45%** — energy-fidelity compounds at 256 tokens × 1408-d with L1 and ~16 s/action CEM; richer language reintroduces multimodality (needs the deferred flow head).

These are calibrated judgments, not measurements, and the sub-risks are correlated; treat the ranges as decision-relevant, not precise forecasts.

---

## 6. Caveats on the landscape
- Several cited arXiv IDs are future-dated relative to a literal reading (2026 prefixes) but resolve to real papers consistent with the June-2026 frame. Re-verify if working from a different time frame.
- **PiJEPA's real title** is "Policy-Guided World Model Planning for Language-Conditioned Visual Navigation" (2603.25981) and it does *not* synthesize a goal latent — don't overstate the overlap.
- Naming-collision: **VLA-JEPA** (2602.10098) is a JEPA-pretraining recipe for VLA policies — a different problem; do not confuse with this project.
- The §10 "closest prior art" list in the original handoff (LBP / Object-Centric WM / WoG) was not re-verified this pass — trust this file's verified set where they overlap.

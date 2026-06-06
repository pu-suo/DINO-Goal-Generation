# torchqmet (vendored)

Vendored verbatim from https://github.com/quasimetric-learning/torch-quasimetric
(Tongzhou Wang), commit fetched 2026-06-05. BSD-3-Clause (see LICENSE).

Why vendored: the vast.ai GPU box syncs Mac->box via `git push/pull` (no extra pip
step), so the IQE/MRN quasimetric heads must live in-repo. Pure PyTorch
(`@torch.jit.script`), no build step. We import it as `third_party.torchqmet`.

We only use `IQE` (default IQE-maxmean) and `MRN`/`MRNFixed` (faster fallback).
`PQE` is unused; its `pqe/cdf_ops` C extension is compiled LAZILY only if PQE is
instantiated (never on import), so importing this package does not require a
compiler. Do not edit these files; treat as a frozen dependency.

# Upstream provenance

This `dino_wm/` directory is a fork of the DINO-WM reference implementation,
extended in place for the Cross-Modal Text-to-Goal Bridge project (Phase 0+).

- **Upstream:** https://github.com/gaoyuezhou/dino_wm.git
- **Forked at commit:** `0a9492fa12044b852ae9e001cc74604b79c8bb0c` ("add checkpoints")
- **Paper:** DINO-WM (Zhou et al., 2024) — https://arxiv.org/abs/2411.04983

The nested upstream `.git` was removed so this lives as a normal subdirectory of
the single project repo (`DINO_Goal_Generation/`). To diff against upstream:

```bash
git clone https://github.com/gaoyuezhou/dino_wm.git /tmp/dino_wm_upstream
git -C /tmp/dino_wm_upstream checkout 0a9492f
diff -ru /tmp/dino_wm_upstream . --exclude .git
```

## What we add (kept isolated from the frozen modules)
All new code lives in the existing package layout and never modifies the frozen
encoder / dynamics / CEM weights. New, additive files only (see project root
`README` / `specs/PHASE_0_PLAN.md`). Edits to existing upstream files are kept
purely additive and called out in commit messages.

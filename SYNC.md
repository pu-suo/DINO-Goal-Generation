# Code sync: Mac (dev) ↔ vast.ai (compute)

Code is generated/edited on the Mac and run on the rented 4090. Transfer is via
**git push/pull** (chosen workflow). This repo is a single git repo: `main` =
pristine upstream `dino_wm` + docs; `phase0` = our work.

## First time — create the remote
```bash
# on the Mac, from the repo root
gh repo create dino-goal-generation --private --source=. --remote=origin   # or set a remote manually:
# git remote add origin git@github.com:<you>/dino-goal-generation.git
git push -u origin main
git push -u origin phase0
```

## On the vast.ai box — clone + setup
```bash
git clone <repo-url> dino_goal && cd dino_goal
git checkout phase0
cd dino_wm && bash scripts/setup_vastai.sh
```

## Iterate (the loop)
```bash
# Mac: edit + commit + push
git add -A && git commit -m "..." && git push origin phase0
# box: pull + run
git pull origin phase0
```

## Notes
- Data, checkpoints, cached latents, plan_outputs are **git-ignored** (see `.gitignore`);
  they live only on the box / OSF, never in git.
- Keep results reproducible: the box pulls a specific commit; record which commit produced
  which numbers (`git rev-parse HEAD`).
- Use spot instances + frequent checkpointing for long jobs; stop idle instances.

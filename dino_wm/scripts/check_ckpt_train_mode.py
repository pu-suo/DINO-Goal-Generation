"""Box check #1 (docs/PLANNING_SPEED_PROFILE.md "Fast config bundle"): was dropout ever
active during planning?

Checkpoints pickle WHOLE module objects, and train.py saves them right after val() --
whose first line is model.eval() -- so the pickled `training` flag is expected to be
False for every module. The plan path never calls .train(), so whatever this prints IS
the mode planning ran in. If predictor=False (expected): the predictor's dropout
(p=0.1) was never active at plan time, the validated SR numbers are dropout-free, and
plan_eval_mode=true is a harmless no-op.

Usage (on the box):
    python scripts/check_ckpt_train_mode.py /workspace/ckpts/outputs/pusht/checkpoints/model_latest.pth
"""
import os
import sys

# Repo root on sys.path BEFORE torch.load: checkpoints pickle WHOLE module objects, so
# unpickling needs to import models.vit etc. (same bootstrap as the sibling scripts).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch import nn

MODULE_KEYS = ["encoder", "predictor", "decoder", "proprio_encoder", "action_encoder"]


def main(path):
    # weights_only=False explicitly: required to unpickle module objects on torch>=2.6
    # (where the default flipped to True); identical behavior on the box's torch 2.3.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    print(f"checkpoint: {path}  (epoch {payload.get('epoch', '?')})")
    verdict = None
    for key in MODULE_KEYS:
        mod = payload.get(key)
        if not isinstance(mod, nn.Module):
            print(f"  {key:16s}: {'absent' if mod is None else type(mod).__name__}")
            continue
        flags = {m.training for m in mod.modules()}
        drops = sorted({m.p for m in mod.modules() if isinstance(m, nn.Dropout) and m.p > 0})
        print(f"  {key:16s}: training={sorted(flags)}  dropout_p>0={drops or 'none'}")
        if key == "predictor":
            verdict = (flags == {False})
    if verdict is None:
        print("VERDICT: no predictor module in this checkpoint?!")
        return 2
    if verdict:
        print("VERDICT: predictor pickled in EVAL mode -> the PREDICTOR's dropout (the only "
              "p>0 dropout in a planning rollout) was never active at plan time; "
              "plan_eval_mode=true is a no-op (expected case).")
        return 0
    print("VERDICT: predictor pickled in TRAIN mode -> its dropout WAS active during "
          "planning; the original PLANNING_SPEED_PROFILE.md RNG analysis applies. Set "
          "plan_eval_mode=true (result-changing) before using the fast config.")
    return 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))

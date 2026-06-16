"""Render top-down sample panels for visual inspection (render quality + pusher hiding).

Saves to dino_wm/analysis/out/langtable_samples/:
  start_panel.png  -- N scenes, each [visible (pusher) | effector-hidden]
  goal_panel.png   -- M oracle pushes, each [start-visible | goal-visible | goal-hidden] + instruction
  frames/...       -- individual PNGs

Run (langtable env):
  /Users/Tom/miniforge3/envs/langtable/bin/python dino_wm/scripts/langtable/lt_render_samples.py
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lt_compat  # noqa: E402
lt_compat.install_tf_agents_shim()

import cv2  # noqa: E402
from language_table.environments import blocks  # noqa: E402
from language_table.environments import language_table  # noqa: E402
from language_table.environments.oracles import push_oracle_rrt_slowdown  # noqa: E402
from language_table.environments.rewards import block2block  # noqa: E402
import lt_render  # noqa: E402

SIZE = 224
SCALE = 2  # upscale for display clarity (shows the actual 224px model input, 2x nearest)
OUT = "/Users/Tom/Active-Projects/DINO_Goal_Generation/dino_wm/analysis/out/langtable_samples"


def up(rgb):
    return cv2.resize(rgb, (SIZE * SCALE, SIZE * SCALE), interpolation=cv2.INTER_NEAREST)


def label(img_bgr, text, y=22):
    cv2.rectangle(img_bgr, (0, 0), (img_bgr.shape[1], 28), (30, 30, 30), -1)
    cv2.putText(img_bgr, text, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return img_bgr


def hcat(imgs, pad=6):
    h = imgs[0].shape[0]
    sep = np.full((h, pad, 3), 255, np.uint8)
    out = []
    for i, im in enumerate(imgs):
        out.append(im)
        if i < len(imgs) - 1:
            out.append(sep)
    return np.concatenate(out, 1)


def vcat(rows, pad=6):
    w = rows[0].shape[1]
    sep = np.full((pad, w, 3), 255, np.uint8)
    out = []
    for i, r in enumerate(rows):
        out.append(r)
        if i < len(rows) - 1:
            out.append(sep)
    return np.concatenate(out, 0)


def decode_instr(obs):
    instr = obs.get("instruction")
    return bytes([c for c in np.asarray(instr).ravel().tolist() if c]).decode("utf-8", "ignore")


def cell(rgb, text):
    bgr = up(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return label(bgr, text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start_scenes", type=int, default=6)
    ap.add_argument("--goal_pairs", type=int, default=4)
    args = ap.parse_args()
    os.makedirs(OUT + "/frames", exist_ok=True)

    env = language_table.LanguageTable(
        block_mode=blocks.LanguageTableBlockVariants.BLOCK_8,
        reward_factory=block2block.BlockToBlockReward, control_frequency=10.0, seed=3)
    aenv = lt_compat.GymToTFAgentsEnv(env)
    oracle = push_oracle_rrt_slowdown.ObstacleOrientedPushOracleBoard2dRRT(aenv, use_ee_planner=True)

    # --- start panel: visible vs hidden across scenes ---
    rows = []
    for s in range(args.start_scenes):
        ts = aenv.reset()
        oracle.reset()
        for _ in range(50):
            if oracle.get_plan(aenv.compute_state()):
                break
            ts = aenv.reset(); oracle.reset()
        vis = lt_render.render_topdown(env, size=SIZE, hide_effector=False)
        hid = lt_render.render_topdown(env, size=SIZE, hide_effector=True)
        cv2.imwrite(f"{OUT}/frames/start{s}_visible.png", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"{OUT}/frames/start{s}_hidden.png", cv2.cvtColor(hid, cv2.COLOR_RGB2BGR))
        rows.append(hcat([cell(vis, f"scene {s}: visible (pusher)"), cell(hid, "effector-hidden")]))
    cv2.imwrite(f"{OUT}/start_panel.png", vcat(rows))
    print(f"saved {OUT}/start_panel.png ({args.start_scenes} scenes)")

    # --- goal panel: oracle push -> start vis | goal vis | goal hidden ---
    rows = []
    done = 0
    attempts = 0
    while done < args.goal_pairs and attempts < args.goal_pairs * 4:
        attempts += 1
        ts = aenv.reset(); oracle.reset()
        for _ in range(50):
            if oracle.get_plan(aenv.compute_state()):
                break
            ts = aenv.reset(); oracle.reset()
        instr = decode_instr(aenv.last_obs)
        start_vis = lt_render.render_topdown(env, size=SIZE, hide_effector=False)
        steps = 0
        while not ts.is_last() and steps < 200:
            ts = aenv.step(oracle.action(ts, ()).action)
            steps += 1
        if not aenv.succeeded:
            continue
        goal_vis = lt_render.render_topdown(env, size=SIZE, hide_effector=False)
        goal_hid = lt_render.render_topdown(env, size=SIZE, hide_effector=True)
        row = hcat([cell(start_vis, f"start: {instr[:34]}"),
                    cell(goal_vis, "goal: visible"),
                    cell(goal_hid, "goal: effector-hidden")])
        rows.append(row)
        cv2.imwrite(f"{OUT}/frames/goal{done}_start.png", cv2.cvtColor(start_vis, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"{OUT}/frames/goal{done}_hidden.png", cv2.cvtColor(goal_hid, cv2.COLOR_RGB2BGR))
        done += 1
    if rows:
        cv2.imwrite(f"{OUT}/goal_panel.png", vcat(rows))
        print(f"saved {OUT}/goal_panel.png ({done} pushes)")
    print("DONE")


if __name__ == "__main__":
    main()

"""Gate-0-fix visual: PushT-style dot pusher. Panels of [arm | dot | clean] + mid-rollout dots."""
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

SIZE, SCALE = 224, 2
OUT = "/Users/Tom/Active-Projects/DINO_Goal_Generation/dino_wm/analysis/out/langtable_samples"


def up(rgb):
    return cv2.resize(rgb, (SIZE * SCALE, SIZE * SCALE), interpolation=cv2.INTER_NEAREST)


def cell(rgb, text):
    bgr = up(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.rectangle(bgr, (0, 0), (bgr.shape[1], 26), (30, 30, 30), -1)
    cv2.putText(bgr, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return bgr


def hcat(imgs, pad=6):
    sep = np.full((imgs[0].shape[0], pad, 3), 255, np.uint8)
    r = []
    for i, im in enumerate(imgs):
        r += [im] + ([sep] if i < len(imgs) - 1 else [])
    return np.concatenate(r, 1)


def vcat(rows, pad=6):
    w = max(r.shape[1] for r in rows)
    rows = [r if r.shape[1] == w else
            np.concatenate([r, np.full((r.shape[0], w - r.shape[1], 3), 255, np.uint8)], 1)
            for r in rows]
    sep = np.full((pad, w, 3), 255, np.uint8)
    r = []
    for i, im in enumerate(rows):
        r += [im] + ([sep] if i < len(rows) - 1 else [])
    return np.concatenate(r, 0)


def ee(aenv):
    return np.asarray(aenv.last_obs["effector_translation"]).ravel()[:2]


def decode_instr(aenv):
    instr = aenv.last_obs.get("instruction")
    return bytes([c for c in np.asarray(instr).ravel().tolist() if c]).decode("utf-8", "ignore")


def main():
    os.makedirs(OUT, exist_ok=True)
    env = language_table.LanguageTable(
        block_mode=blocks.LanguageTableBlockVariants.BLOCK_8,
        reward_factory=block2block.BlockToBlockReward, control_frequency=10.0, seed=5)
    aenv = lt_compat.GymToTFAgentsEnv(env)
    oracle = push_oracle_rrt_slowdown.ObstacleOrientedPushOracleBoard2dRRT(aenv, use_ee_planner=True)

    # panel 1: arm vs dot vs clean across scenes
    rows = []
    for s in range(4):
        ts = aenv.reset(); oracle.reset()
        for _ in range(50):
            if oracle.get_plan(aenv.compute_state()):
                break
            ts = aenv.reset(); oracle.reset()
        e = ee(aenv)
        arm = lt_render.render_topdown(env, SIZE, mode="arm")
        dot = lt_render.render_topdown(env, SIZE, mode="dot", ee_xy=e)
        clean = lt_render.render_topdown(env, SIZE, mode="clean")
        rows.append(hcat([cell(arm, f"{s}: ARM (debug, occludes)"),
                          cell(dot, "DOT (start/rollout)"),
                          cell(clean, "CLEAN (goal)")]))
    cv2.imwrite(f"{OUT}/dot_modes_panel.png", vcat(rows))
    print(f"saved {OUT}/dot_modes_panel.png")

    # panel 2: mid-rollout dot frames near contact (oracle stepped)
    rows = []
    for s in range(3):
        ts = aenv.reset(); oracle.reset()
        for _ in range(50):
            if oracle.get_plan(aenv.compute_state()):
                break
            ts = aenv.reset(); oracle.reset()
        shots = []
        step = 0
        targets = [0, 20, 40]
        while not ts.is_last() and step <= max(targets):
            if step in targets:
                shots.append(cell(lt_render.render_topdown(env, SIZE, mode="dot", ee_xy=ee(aenv)),
                                  f"rollout t={step}"))
            ts = aenv.step(oracle.action(ts, ()).action); step += 1
        if shots:
            rows.append(hcat(shots))
    if rows:
        cv2.imwrite(f"{OUT}/dot_rollout_panel.png", vcat(rows))
        print(f"saved {OUT}/dot_rollout_panel.png")

    # panel 3: relational pipeline with new render -> [start (dot) | goal (clean)]
    rows = []
    done = 0
    att = 0
    while done < 4 and att < 16:
        att += 1
        ts = aenv.reset(); oracle.reset()
        for _ in range(50):
            if oracle.get_plan(aenv.compute_state()):
                break
            ts = aenv.reset(); oracle.reset()
        instr = decode_instr(aenv)
        start_dot = lt_render.render_topdown(env, SIZE, mode="dot", ee_xy=ee(aenv))
        step = 0
        while not ts.is_last() and step < 200:
            ts = aenv.step(oracle.action(ts, ()).action); step += 1
        if not aenv.succeeded:
            continue
        goal_clean = lt_render.render_topdown(env, SIZE, mode="clean")
        rows.append(hcat([cell(start_dot, f"start: {instr[:30]}"), cell(goal_clean, "goal (clean, no pusher)")]))
        done += 1
    if rows:
        cv2.imwrite(f"{OUT}/dot_goal_pairs_panel.png", vcat(rows))
        print(f"saved {OUT}/dot_goal_pairs_panel.png")
    print("DONE")


if __name__ == "__main__":
    main()

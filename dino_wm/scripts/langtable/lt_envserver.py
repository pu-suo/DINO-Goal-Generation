"""Language-Table sim SERVER (langtable conda env) for the closed-loop planner.

Holds a live LanguageTable block2block env and answers commands from the planner (base env)
over a socket. The planner cannot import pybullet/language_table; this process cannot import
torch. So the sim lives here and ships rendered DOT frames (224,224,3 uint8) + ee + GT block_xy
across the wire; the planner encodes them with DINOv2 on the GPU.

Frame/coord contract is IDENTICAL to lt_dump_traj.py + lt_render.py (so live latents match the
cached training distribution): mode='dot', SIZE=224, FIXED_8 block order, half_extent/center from
lt_render. Valid init = oracle.get_plan succeeds (same solvable-init distribution as the corpus).

Protocol (lt_ipc, length-prefixed pickle):
  {"cmd":"reset","seed":int?}     -> {ok, frame, ee, block_xy, blocks, start_block, target_block,
                                      instruction, success, done}
  {"cmd":"step","actions":[[dx,dy],...]} -> {ok, frame, ee, block_xy, reward, success, done}
       (executes the list of ENV actions in sequence = one model-step of `frameskip` env actions;
        renders/returns ONLY the final frame; stops early if the episode terminates)
  {"cmd":"close"} -> exits

Run (langtable env): python lt_envserver.py --port <P> [--seed 0] [--size 224]
The planner spawns this; it is not launched by hand.
"""
import argparse
import os
import socket
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lt_compat  # noqa: E402
lt_compat.install_tf_agents_shim()

from language_table.environments import blocks  # noqa: E402
from language_table.environments import language_table  # noqa: E402
from language_table.environments.oracles import push_oracle_rrt_slowdown  # noqa: E402
from language_table.environments.rewards import block2block  # noqa: E402
import lt_ipc  # noqa: E402
import lt_render  # noqa: E402

ORDER = list(blocks.FIXED_8_COMBINATION)


def state_xy(env):
    st = env.compute_state()
    return np.array([np.asarray(st[f"block_{b}_translation"]).ravel()[:2] for b in ORDER], np.float32)


def ee_xy(aenv):
    return np.asarray(aenv.last_obs["effector_translation"]).ravel()[:2].astype(np.float32)


def decode_instr(aenv):
    instr = aenv.last_obs.get("instruction")
    if instr is None:
        return ""
    return bytes([c for c in np.asarray(instr).ravel().tolist() if c]).decode("utf-8", "ignore")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--control_freq", type=float, default=10.0)
    ap.add_argument("--max_init_tries", type=int, default=50)
    a = ap.parse_args()

    env = language_table.LanguageTable(
        block_mode=blocks.LanguageTableBlockVariants.BLOCK_8,
        reward_factory=block2block.BlockToBlockReward,
        control_frequency=a.control_freq, seed=a.seed)
    aenv = lt_compat.GymToTFAgentsEnv(env)
    oracle = push_oracle_rrt_slowdown.ObstacleOrientedPushOracleBoard2dRRT(aenv, use_ee_planner=True)
    state = {"ts": None, "done": True}

    def render():
        return lt_render.render_topdown(env, a.size, mode="dot", ee_xy=ee_xy(aenv))

    def do_reset(no_terminate=False):
        ts = None
        for _ in range(a.max_init_tries):
            ts = aenv.reset()
            oracle.reset()
            try:
                if oracle.get_plan(aenv.compute_state()):
                    break
            except Exception:  # noqa: BLE001
                continue
        # L4-only harness: suppress the reward's episode-termination so the planner controls a
        # MULTI-STEP chain (the env otherwise ends when its OWN random pair succeeds; physics/render
        # unchanged; the frozen WM/readout and the GT-distance metric are untouched).
        rc = env._reward_calculator
        if no_terminate and not getattr(rc, "_noterm_wrapped", False):
            _orig_reward = rc.reward
            rc.reward = lambda s, _o=_orig_reward: (_o(s)[0], False)
            rc._noterm_wrapped = True
        state["ts"] = ts
        state["done"] = bool(ts.is_last())
        return {
            "ok": True,
            "frame": render(),
            "ee": ee_xy(aenv),
            "block_xy": state_xy(env),
            "blocks": list(ORDER),
            "start_block": str(getattr(rc, "_start_block", "")),
            "target_block": str(getattr(rc, "_target_block", "")),
            "instruction": decode_instr(aenv),
            "success": bool(aenv.succeeded),
            "done": state["done"],
            "half_extent": float(lt_render.DEFAULT_HALF_EXTENT),
            "center": list(lt_render.CENTER),
        }

    def do_step(actions):
        if state["done"]:
            return {"ok": False, "err": "episode done; reset first"}
        reward = 0.0
        for act in actions:
            ts = aenv.step(np.asarray(act, np.float32))
            state["ts"] = ts
            reward = float(ts.reward) if ts.reward is not None else 0.0
            if ts.is_last():
                state["done"] = True
                break
        return {
            "ok": True,
            "frame": render(),
            "ee": ee_xy(aenv),
            "block_xy": state_xy(env),
            "reward": reward,
            "success": bool(aenv.succeeded),
            "done": state["done"],
        }

    # connect back to the planner (server side); retry while the planner binds/accepts.
    sock = None
    for _ in range(100):
        try:
            sock = socket.create_connection((a.host, a.port), timeout=5)
            break
        except OSError:
            time.sleep(0.1)
    if sock is None:
        print("envserver: could not connect to planner", file=sys.stderr)
        sys.exit(1)
    # create_connection leaves a 5s timeout on the socket; the server then blocks here waiting for the
    # planner's NEXT command, and that gap = the planner's CEM-plan time. Under concurrent GPU load a
    # plan can exceed 5s -> recv() times out -> the server exits and the planner sees a dropped conn.
    # Use a generous command-wait timeout (still finite so a truly dead planner is detected).
    sock.settimeout(1200)
    print(f"envserver: connected to {a.host}:{a.port}", file=sys.stderr)

    while True:
        msg = lt_ipc.recv(sock)
        if msg is None or msg.get("cmd") == "close":
            break
        cmd = msg.get("cmd")
        try:
            if cmd == "reset":
                if msg.get("seed") is not None:
                    try:
                        env.seed(int(msg["seed"]))
                    except Exception:  # noqa: BLE001  (older gym envs vary; seq resets still vary layout)
                        pass
                resp = do_reset(no_terminate=bool(msg.get("no_terminate", False)))
            elif cmd == "step":
                resp = do_step(msg["actions"])
            elif cmd == "ping":
                resp = {"ok": True, "pong": True}
            else:
                resp = {"ok": False, "err": f"unknown cmd {cmd}"}
        except Exception as e:  # noqa: BLE001
            import traceback
            resp = {"ok": False, "err": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()}
        lt_ipc.send(sock, resp)
    sock.close()
    print("envserver: closed", file=sys.stderr)


if __name__ == "__main__":
    main()

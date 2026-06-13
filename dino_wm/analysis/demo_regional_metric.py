"""Part-0 sanity demo for the PROPOSED regional success metric.

Runs a battery of constructed cases through BOTH the proposed regional metric
(metrics/regional_success.py) and the stock single-point gate (a faithful
reimplementation of PushTWrapper.eval_state, pose_only_success=True) and prints
a side-by-side verdict table, then renders the 3x3 partition with the goal T and
each test T overlaid.

  /Users/Tom/miniforge3/envs/dino_wm_dev/bin/python analysis/demo_regional_metric.py
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, Rectangle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metrics.regional_success import (
    regional_success, block_cell, region_name, REGION_BOUNDS, REGION_NCELLS, ANG_TOL,
)

# --- stock single-point gate (faithful copy of PushTWrapper.eval_state) -------
def stock_success(goal_state, cur_state):
    pos_diff = np.linalg.norm(goal_state[2:4] - cur_state[2:4])   # pose_only_success
    ad = np.abs(goal_state[4] - cur_state[4])
    ad = np.minimum(ad, 2 * np.pi - ad)
    return bool(pos_diff < 20 and ad < np.pi / 9), float(pos_diff), float(np.degrees(ad))

# --- inline T geometry (matches multicolor_common.tee_world_vertices) ---------
TS, L = 30, 4
RECT1 = np.array([(-L*TS/2, TS), (L*TS/2, TS), (L*TS/2, 0), (-L*TS/2, 0)], float)
RECT2 = np.array([(-TS/2, TS), (-TS/2, L*TS), (TS/2, L*TS), (TS/2, TS)], float)

def tee_polys(pose):
    x, y, th = pose[0], pose[1], pose[4] if len(pose) > 4 else pose[2]
    c, s = np.cos(th), np.sin(th)
    R = np.array([[c, -s], [s, c]])
    return [r @ R.T + np.array([x, y]) for r in (RECT1, RECT2)]

def st(ax, ay, bx, by, th):
    return np.array([ax, ay, bx, by, th], float)

# --- case battery: (name, goal_state, cur_state, expectation note) ------------
GOAL = st(250, 250, 250, 250, 0.30)   # block origin (250,250) -> cell (1,1) center-middle
CASES = [
    ("1 exact hit",            st(0,0, 255, 248, 0.32), "true point-hit: BOTH succeed"),
    ("2 in-region near-miss",  st(0,0, 290, 210, 0.40), "56px off, same cell, ang ok -> REGIONAL succeeds, single-point FAILS (the whole point)"),
    ("3 position miss",        st(0,0, 350, 250, 0.30), "different cell -> BOTH fail (pos)"),
    ("4 orientation miss",     st(0,0, 255, 250, 1.10), "same cell, 46deg off -> BOTH fail (ang)"),
]

print("=" * 108)
print("goal block origin (%.0f,%.0f) angle %.1fdeg -> region %s  cell %s" % (
    GOAL[2], GOAL[3], np.degrees(GOAL[4]), region_name(block_cell(GOAL[2:4])), block_cell(GOAL[2:4])))
print("ANG_TOL = %.0f deg | partition %dx%d over %s" % (
    np.degrees(ANG_TOL), REGION_NCELLS, REGION_NCELLS, REGION_BOUNDS))
print("=" * 108)
hdr = "%-22s | %-26s | %-24s | %s"
print(hdr % ("case", "REGIONAL (proposed)", "single-point (stock)", "expectation met?"))
print("-" * 108)
results = []
for name, cur, note in CASES:
    r = regional_success(GOAL, cur)
    s_ok, s_pos, s_ang = stock_success(GOAL, cur)
    reg_str = "%s (cell %s, %s; ang %.0fdeg %s)" % (
        "PASS" if r["success"] else "fail",
        r["cur_region"], "pos OK" if r["pos_ok"] else "pos X",
        r["ang_diff_deg"], "ang OK" if r["ang_ok"] else "ang X")
    stk_str = "%s (%.0fpx, %.0fdeg)" % ("PASS" if s_ok else "fail", s_pos, s_ang)
    print(hdr % (name, ("PASS" if r["success"] else "fail"), ("PASS" if s_ok else "fail"), note))
    print("%-22s |   %-24s |   %-22s |" % ("", reg_str, stk_str))
    results.append((name, cur, r, (s_ok, s_pos, s_ang), note))
print("-" * 108)

# --- dedicated boundary-straddle demo (the honest regional limitation) --------
# Goal sits 5px from the x=200 cell edge; achieved is 10px away but ACROSS it.
g_b = st(0, 0, 195, 250, 0.30)   # cell (0,1) "middle-left"
c_b = st(0, 0, 205, 250, 0.30)   # cell (1,1) "middle-center"
rb = regional_success(g_b, c_b)
sb_ok, sb_pos, sb_ang = stock_success(g_b, c_b)
print("BOUNDARY STRADDLE (separate goal near a cell edge):")
print("  goal (195,250) cell %s  vs  cur (205,250) cell %s  -> %dpx apart" % (
    rb["goal_cell"], rb["cur_cell"], sb_pos))
print("  REGIONAL=%s (different cell)   single-point=%s (%dpx<20)" % (
    "fail" if not rb["success"] else "PASS", "PASS" if sb_ok else "fail", sb_pos))
print("  -> near a boundary the regional metric is STRICTER, not looser. Expected;")
print("     mitigated by coarse cells (goals are interior most of the time). A soft")
print("     margin / overlapping regions is a review option if this matters.")
print("-" * 108)

# --- visual ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 8))
lo, hi = REGION_BOUNDS
step = (hi - lo) / REGION_NCELLS
# shade the goal cell
gc = block_cell(GOAL[2:4])
ax.add_patch(Rectangle((lo + gc[0]*step, lo + gc[1]*step), step, step,
                        facecolor="gold", alpha=0.18, zorder=0))
# grid lines + region names
for k in range(REGION_NCELLS + 1):
    ax.axvline(lo + k*step, color="0.6", lw=1, ls="--")
    ax.axhline(lo + k*step, color="0.6", lw=1, ls="--")
for cc in range(REGION_NCELLS):
    for rr in range(REGION_NCELLS):
        ax.text(lo + (cc+0.5)*step, lo + (rr+0.5)*step, region_name((cc, rr)),
                ha="center", va="center", color="0.7", fontsize=8, zorder=0)

def draw_tee(ax, pose, color, lw, label):
    for poly in tee_polys(pose):
        ax.add_patch(MplPolygon(poly, closed=True, fill=False, edgecolor=color, lw=lw))
    ax.plot(pose[2], pose[3], "o", color=color, ms=5)
    ax.plot([], [], "-", color=color, lw=2, label=label)

draw_tee(ax, GOAL, "green", 3, "GOAL (language target)")
palette = ["tab:blue", "tab:purple", "tab:red", "tab:orange", "tab:brown"]
for (name, cur, r, stk, note), col in zip(results, palette):
    tag = "%s | reg=%s stock=%s" % (name.split(" ", 1)[1], "PASS" if r["success"] else "fail",
                                    "PASS" if stk[0] else "fail")
    draw_tee(ax, cur, col, 1.6, tag)

ax.set_xlim(50, 450); ax.set_ylim(50, 450)
ax.set_aspect("equal"); ax.set_title("Proposed regional metric vs stock single-point gate\n"
              "(goal cell shaded; 3x3 partition = REGION_BOUNDS)")
ax.set_xlabel("sim-x (512)"); ax.set_ylabel("sim-y (512)  [y-up here; render convention TBD in Part 1]")
ax.legend(loc="upper left", fontsize=7, framealpha=0.9)
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "analysis_outputs", "regional_metric_demo.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=110, bbox_inches="tight")
print("saved visual -> %s" % out)

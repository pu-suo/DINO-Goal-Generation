"""
Decorrelated layout sampler for multi-color PushT (host-side, pure numpy).

A "layout" fully specifies one episode's *physical* scene (independent of the
instruction phrasing):

    {
      'init_state'  : (7,) or (5,)  [agent_x, agent_y, block_x, block_y, angle, (vx, vy)]
      'targets'     : [ {'color','rgb','pose':(x,y,theta),'bin':int}, ... ]  (N of them)
      'active_idx'  : int           # which target is named (the goal)
      'active_color': str
      'goal_pose'   : (3,)          # == targets[active_idx]['pose']
      'instruction' : str
      'template_id' : int
      'seed'        : int
    }

Two independence guarantees this module enforces:

* **Decorrelation (Phase 0.1):** `active_idx` is drawn uniformly over the N
  targets, *independent* of the block's start pose. Hence P(named == nearest
  target) == 1/N (chance). `nearest_target_predicts_named` verifies this.

* **Held-out color-location split (Phase 0.2 scaffold):** the workspace is
  binned into an `n_bins x n_bins` grid; a (color, bin) pairing is a "combo".
  `make_combo_split` partitions combos into train/test. In TRAIN layouts every
  target uses a TRAIN combo (so a held-out pairing never appears, even as a
  distractor). In TEST layouts the *active* target uses a TEST combo while
  distractors stay in TRAIN combos, isolating the novelty to the named pairing.
"""

import numpy as np

from .instructions import sample_instruction
from .multicolor_common import get_palette

DEFAULT_POS_RANGE = (120.0, 392.0)   # inset from the 512 frame so T-decals stay visible
DEFAULT_MIN_SEP = 130.0              # min center separation between targets (px, 512 space);
                                     # targets are FILLED T's (~120px span) so this keeps them
                                     # from fully occluding each other (no target is drawn on top).


# --- workspace binning + combo splits ----------------------------------------
def workspace_bin(pose, n_bins=3, pos_range=DEFAULT_POS_RANGE):
    """Flatten (x, y) into a single bin id in [0, n_bins**2)."""
    lo, hi = pos_range
    span = max(hi - lo, 1e-6)
    bx = int(np.clip((pose[0] - lo) / span * n_bins, 0, n_bins - 1))
    by = int(np.clip((pose[1] - lo) / span * n_bins, 0, n_bins - 1))
    return by * n_bins + bx


def all_color_location_combos(n_targets, n_bins=3):
    palette = get_palette(n_targets)
    names = [c for c, _ in palette]
    return [(c, b) for c in names for b in range(n_bins * n_bins)]


def make_combo_split(n_targets, n_bins=3, heldout_frac=0.2, seed=0):
    """Partition (color, bin) combos into (train_combos, test_combos) sets.

    Every color and every bin still appears in TRAIN (we hold out *pairings*,
    not whole colors/locations), so this is true compositional recombination.
    """
    rng = np.random.RandomState(seed)
    combos = all_color_location_combos(n_targets, n_bins)
    palette = get_palette(n_targets)
    names = [c for c, _ in palette]
    n_bin_ids = n_bins * n_bins

    combo_set = set(combos)
    n_test = int(round(len(combos) * heldout_frac))
    # Greedily pick held-out combos but never remove the last combo for a given
    # color or bin (keep every color and every bin represented in train).
    order = list(combos)
    rng.shuffle(order)
    test = set()
    color_left = {c: n_bin_ids for c in names}
    bin_left = {b: n_targets for b in range(n_bin_ids)}
    for (c, b) in order:
        if len(test) >= n_test:
            break
        if color_left[c] > 1 and bin_left[b] > 1:
            test.add((c, b))
            color_left[c] -= 1
            bin_left[b] -= 1
    train = combo_set - test
    return train, test


# --- core samplers ------------------------------------------------------------
def _sample_init_state(rng, with_velocity):
    """Match PushTEnv.reset()'s init distribution exactly (in-distribution)."""
    base = [
        rng.randint(50, 450),
        rng.randint(50, 450),
        rng.randint(100, 400),
        rng.randint(100, 400),
        rng.randn() * 2 * np.pi - np.pi,
    ]
    if with_velocity:
        base += [0.0, 0.0]
    return np.array(base, dtype=np.float64)


def _sample_positions(rng, n, pos_range, min_sep, max_tries=2000):
    pts = []
    tries = 0
    while len(pts) < n and tries < max_tries:
        tries += 1
        p = rng.uniform(pos_range[0], pos_range[1], size=2)
        if all(np.linalg.norm(p - q) >= min_sep for q in pts):
            pts.append(p)
    if len(pts) < n:
        return None
    return np.stack(pts)


def _assign_colors(bins, names, active_slot, allowed, active_allowed, rng):
    """Distinct color per slot s.t. (color, bin) is in the allowed set per slot.

    Backtracking; allowed/active_allowed are None (no constraint) or a set of
    (color, bin) combos. Returns list[str] or None if infeasible.
    """
    n = len(bins)
    order = [active_slot] + [i for i in range(n) if i != active_slot]
    used, result = set(), [None] * n

    def feasible(slot, color):
        ok = active_allowed if slot == active_slot else allowed
        return ok is None or (color, bins[slot]) in ok

    def bt(k):
        if k == n:
            return True
        slot = order[k]
        cand = [c for c in names if c not in used and feasible(slot, c)]
        rng.shuffle(cand)
        for c in cand:
            used.add(c)
            result[slot] = c
            if bt(k + 1):
                return True
            used.discard(c)
            result[slot] = None
        return False

    return result if bt(0) else None


def sample_layout(
    seed,
    n_targets=4,
    with_velocity=True,
    pos_range=DEFAULT_POS_RANGE,
    min_sep=DEFAULT_MIN_SEP,
    n_bins=3,
    allowed_combos=None,
    active_combos=None,
    instruction_held_out=False,
    max_goal_dist=None,
    max_goal_angle=None,
    max_resamples=200,
):
    """Sample one decorrelated, optionally split-constrained layout.

    allowed_combos: set of (color, bin) every target must satisfy (or None).
    active_combos:  set the *active* target must satisfy (defaults to allowed).
    max_goal_dist:  if set, the NAMED target is restricted to within this many px
                    of the block start, i.e. a short reachable push. This is for
                    easy/fast oracle testing ONLY -- it intentionally BREAKS
                    decorrelation (the named target becomes the nearest one), so
                    never use it for g's dataset or the headline held-out eval.
    max_goal_angle: if set (radians), the NAMED target's orientation is forced to
                    within this much of the block's START angle, i.e. ~no rotation
                    required. Pair with a small max_goal_dist for a TRIVIALLY
                    reachable goal (sanity-checking the frozen stack). Like
                    max_goal_dist this is a diagnostic crutch -- not for g/headline.
    """
    rng = np.random.RandomState(seed)
    palette = get_palette(n_targets)
    names = [c for c, _ in palette]
    rgb_of = dict(palette)

    for _ in range(max_resamples):
        init_state = _sample_init_state(rng, with_velocity)
        positions = _sample_positions(rng, n_targets, pos_range, min_sep)
        if positions is None:
            continue
        angles = rng.uniform(0.0, 2 * np.pi, size=n_targets)
        bins = [workspace_bin(p, n_bins, pos_range) for p in positions]

        # Active-target selection. Default: uniform over slots, independent of the
        # block (decorrelated; P(named==nearest)=1/N). With max_goal_dist set, pick
        # the active target from those within reach of the block -> short push.
        if max_goal_dist is None:
            active_slot = int(rng.randint(n_targets))
        else:
            d_block = np.linalg.norm(positions - init_state[2:4], axis=1)
            cand = np.where(d_block <= max_goal_dist)[0]
            if len(cand) == 0:
                continue  # no target close enough; resample the layout
            active_slot = int(rng.choice(cand))

        # Optional: pin the NAMED target's orientation near the block's start angle
        # so the push needs ~no rotation (trivially-reachable diagnostic goal).
        if max_goal_angle is not None:
            angles[active_slot] = init_state[4] + rng.uniform(-max_goal_angle, max_goal_angle)

        colors = _assign_colors(
            bins, names, active_slot, allowed_combos,
            active_combos if active_combos is not None else allowed_combos, rng,
        )
        if colors is None:
            continue

        targets = [
            {
                "color": colors[i],
                "rgb": rgb_of[colors[i]],
                "pose": np.array([positions[i, 0], positions[i, 1], angles[i]], dtype=np.float64),
                "bin": bins[i],
            }
            for i in range(n_targets)
        ]
        active_color = colors[active_slot]
        instruction, template_id = sample_instruction(
            active_color, rng, held_out=instruction_held_out
        )
        return {
            "init_state": init_state,
            "targets": targets,
            "active_idx": active_slot,
            "active_color": active_color,
            "goal_pose": targets[active_slot]["pose"].copy(),
            "instruction": instruction,
            "template_id": template_id,
            "seed": int(seed),
        }

    raise RuntimeError(
        f"sample_layout failed after {max_resamples} resamples "
        f"(n_targets={n_targets}, min_sep={min_sep}, constrained={allowed_combos is not None})."
    )


# --- decorrelation diagnostic -------------------------------------------------
def nearest_target_predicts_named(n_samples=4000, n_targets=4, seed=0, **kw):
    """Empirical P(named target == nearest target to the block's start centroid).

    Should be ~1/n_targets if the active target is decorrelated from geometry.
    Returns (rate, chance, n_samples).
    """
    from .multicolor_common import tee_centroid_offset
    cx, cy = tee_centroid_offset()
    hits = 0
    for i in range(n_samples):
        lay = sample_layout(seed * 100003 + i, n_targets=n_targets, **kw)
        s = lay["init_state"]
        # block visual centroid in world coords
        th = s[4]
        c, sn = np.cos(th), np.sin(th)
        bx = s[2] + cx * c - cy * sn
        by = s[3] + cx * sn + cy * c
        dists = [np.linalg.norm(np.array([bx, by]) - t["pose"][:2]) for t in lay["targets"]]
        if int(np.argmin(dists)) == lay["active_idx"]:
            hits += 1
    return hits / n_samples, 1.0 / n_targets, n_samples

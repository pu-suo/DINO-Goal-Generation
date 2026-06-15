"""Option B (Part 1): rotation-centric qualitative command + the SHARED rotation-
bucket definition.

Part D verdict: rotation is language-load-bearing (start-alone R2 0.47 -> +correct
command 0.72 -> +swapped command 0.17); displacement direction is scene-determined
(leaked, R2 0.72) and is NOT claimed as language-grounded. So the command carries
ROTATION ONLY: sign + a coarse magnitude band. No numeric angle (a numeric angle is
a closed-form control command and would gut the grounding claim); no direction; no
displacement magnitude.

This module owns the bucket definition. metrics/rotation_goal_success.py imports it,
so the command text (here) and the success metric (there) provably share one bucket
partition -- they cannot drift.

Bands are data-informed (Part A survivors: |drot| p50=6deg, p90=10.5deg; within-
bucket rot-std ~1.6deg, so >=4deg bands are well-resolved) and sit below the pi/9
(20deg) success gate. Sign convention matches the repo's make_language: positive
signed Delta(angle) -> "clockwise".
"""
import numpy as np

ROT_MAG_EDGES_DEG = (3.0, 8.0)              # bands: none[0,3) slight[3,8) moderate[8,inf)
ROT_MAG_NAMES = ("none", "slight", "moderate")
CW_NAMES = {1: "clockwise", -1: "counterclockwise"}


def wrap_deg(a):
    """Wrap degrees to (-180, 180]."""
    return (np.asarray(a, dtype=float) + 180.0) % 360.0 - 180.0


def signed_drot_deg(start_angle_rad, goal_angle_rad):
    """Signed relative rotation start->goal in DEG, wrapped (-180,180]. Inputs in rad."""
    return float(wrap_deg(np.degrees(float(goal_angle_rad) - float(start_angle_rad))))


def rotation_bucket(drot_deg):
    """Map a signed relative rotation (deg) to its (sign, mag_band) bucket.
    mag_band: 0=none, 1=slight, 2=moderate. The 'none' band is sign-agnostic (sign=0)."""
    mag = abs(float(drot_deg))
    band = int(np.digitize(mag, ROT_MAG_EDGES_DEG))     # 0,1,2
    sign = 0 if band == 0 else (1 if drot_deg >= 0 else -1)
    return (sign, band)


def rotation_command_text(drot_deg):
    """Qualitative rotation phrase. No numeric angle, no direction, no displacement."""
    sign, band = rotation_bucket(drot_deg)
    if band == 0:
        return "without rotating it"
    return f"rotating it {ROT_MAG_NAMES[band]} {CW_NAMES[sign]}"


def rotation_in_bucket(achieved_drot_deg, commanded_bucket):
    """Bucket membership: achieved relative rotation lands in the SAME (sign, band)
    bucket as the command. Strict (matches Part 2's coarse-band success)."""
    return rotation_bucket(achieved_drot_deg) == tuple(commanded_bucket)


def all_buckets():
    """The fixed, small command set: none + (slight,moderate) x (CW,CCW) = 5."""
    out = [(0, 0)]
    for band in (1, 2):
        for sign in (1, -1):
            out.append((sign, band))
    return out


def bucket_name(bucket):
    sign, band = bucket
    if band == 0:
        return "none"
    return f"{ROT_MAG_NAMES[band]}_{('cw' if sign == 1 else 'ccw')}"

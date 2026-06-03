"""
Shared utilities for the Phase 0.3 representation probes.

Both probes consume the CACHED start/goal latents (196x384) from
scripts/cache_latents.py plus the per-episode labels, so they are cheap (no
re-encoding). The only subtlety is mapping the sim (512) coordinate frame to the
DINOv2 input grid: the env renders at 512, cv2.resize -> render_size (224),
default_transform CenterCrop(224) (no-op), then the world model Resize(196). So
sim(x,y) -> 196-image (x*196/512, y*196/512); patch (row,col) of size 14 ->
ViT token index row*14 + col (row-major).
"""
import pickle
from pathlib import Path

import numpy as np
import cv2
import torch

from env.pusht.multicolor_common import get_palette, tee_world_vertices

SIM = 512
IMG = 196
PATCH = 14
GRID = IMG // PATCH  # 14
N_TOKENS = GRID * GRID  # 196
AGENT_RADIUS_SIM = 15


def color_id_map(n_targets):
    return {name: i for i, (name, _) in enumerate(get_palette(n_targets))}


def _sim_to_img(pts):
    return np.asarray(pts, dtype=np.float32) * (IMG / SIM)


def rasterize_patch_labels(layout, init_state, n_targets, outline_thickness_sim=7,
                           min_pixels=3, occ_frac=0.35):
    """Per-patch color label for one frame's DINOv2 grid.

    Returns int array (N_TOKENS,) with values: color_id in [0,n_targets), -1 for
    background, -2 for patches occluded by the block/agent (DINO features there
    reflect the manipulator/block, not the decal, so we drop them).
    """
    cmap = color_id_map(n_targets)
    color_img = np.full((IMG, IMG), -1, dtype=np.int32)
    th = max(1, round(outline_thickness_sim * IMG / SIM))
    for tgt in layout["targets"]:
        cid = cmap[tgt["color"]]
        m = np.zeros((IMG, IMG), np.uint8)
        for rect in tee_world_vertices(tgt["pose"]):
            cv2.polylines(m, [_sim_to_img(rect).round().astype(np.int32)], True, 255, th)
        color_img[m > 0] = cid

    # occluder: filled block-T at its start pose + agent circle
    occ = np.zeros((IMG, IMG), np.uint8)
    for rect in tee_world_vertices(np.asarray(init_state)[2:5]):
        cv2.fillPoly(occ, [_sim_to_img(rect).round().astype(np.int32)], 255)
    ax, ay = _sim_to_img(np.asarray(init_state)[:2])
    cv2.circle(occ, (int(round(ax)), int(round(ay))), max(1, round(AGENT_RADIUS_SIM * IMG / SIM)), 255, -1)

    labels = np.full(N_TOKENS, -1, dtype=np.int32)
    for r in range(GRID):
        for c in range(GRID):
            tok = r * GRID + c
            cregion = color_img[r * PATCH:(r + 1) * PATCH, c * PATCH:(c + 1) * PATCH]
            oregion = occ[r * PATCH:(r + 1) * PATCH, c * PATCH:(c + 1) * PATCH]
            if (oregion > 0).mean() > occ_frac:
                labels[tok] = -2
                continue
            present = cregion[cregion >= 0]
            if present.size:
                vals, counts = np.unique(present, return_counts=True)
                if counts.max() >= min_pixels:
                    labels[tok] = int(vals[counts.argmax()])
    return labels


def load_probe_data(data_path, latent_dir, split):
    start = torch.load(Path(latent_dir) / split / "start_latents.pth").float()
    goal = torch.load(Path(latent_dir) / split / "goal_latents.pth").float()
    with open(Path(data_path) / split / "labels.pkl", "rb") as f:
        labels = pickle.load(f)
    init_states = np.stack([np.asarray(l["init_state"]) for l in labels]).astype(np.float32)
    goal_poses = np.stack([np.asarray(l["goal_pose"]) for l in labels]).astype(np.float32)
    return {"start": start, "goal": goal, "labels": labels,
            "init_states": init_states, "goal_poses": goal_poses}


def build_grounding_dataset(data, n_targets, outline_thickness_sim=7, bg_per_frame=8, seed=0):
    """(features, color_ids) over START frames; color_ids in [0,n_targets] (last = background)."""
    rng = np.random.RandomState(seed)
    start = data["start"]
    feats, ys, ep_ids = [], [], []
    for ep, lab in enumerate(data["labels"]):
        layout = _label_to_layout(lab)
        plabels = rasterize_patch_labels(layout, data["init_states"][ep], n_targets, outline_thickness_sim)
        colored = np.where(plabels >= 0)[0]
        bg = np.where(plabels == -1)[0]
        if len(bg) > bg_per_frame:
            bg = rng.choice(bg, bg_per_frame, replace=False)
        for tok in colored:
            feats.append(start[ep, tok]); ys.append(int(plabels[tok])); ep_ids.append(ep)
        for tok in bg:
            feats.append(start[ep, tok]); ys.append(n_targets); ep_ids.append(ep)  # background class
    return torch.stack(feats), torch.tensor(ys), np.array(ep_ids)


def build_pose_dataset(data):
    """(grid_features (M,196,384), block_pose (M,3)) from start+goal frames."""
    X = torch.cat([data["start"], data["goal"]], dim=0)
    start_pose = data["init_states"][:, 2:5]
    pose = np.concatenate([start_pose, data["goal_poses"]], axis=0)
    ep = np.concatenate([np.arange(len(data["labels"]))] * 2)
    return X, torch.tensor(pose, dtype=torch.float32), ep


def _label_to_layout(lab):
    poses = np.asarray(lab["target_poses"])
    return {"targets": [{"color": c, "pose": poses[i]} for i, c in enumerate(lab["target_colors"])]}


def episode_split(ep_ids, n_eps, frac=0.2, seed=0):
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_eps)
    n_test = max(1, int(round(n_eps * frac)))
    test_eps = set(perm[:n_test].tolist())
    is_test = np.array([e in test_eps for e in ep_ids])
    return ~is_test, is_test

#!/usr/bin/env python3
import os
import json
from glob import glob
from tqdm import tqdm
import argparse

import torch
import numpy as np


# ─────────────────────────────────────────────────────────────
# 1) Utility: Load VAD from a .pt file
# ─────────────────────────────────────────────────────────────
def load_vad(path_pt: str) -> np.ndarray:
    """
    Load a VAD vector from a .pt file and rescale from [0, 1] to [-1, 1].

    Returns:
        vad: numpy array of shape (3,) for a single VAD vector.
    """
    if not os.path.isfile(path_pt):
        raise FileNotFoundError(f"No VAD .pt file at {path_pt}")

    vad = torch.load(path_pt, map_location="cpu")
    if isinstance(vad, torch.Tensor):
        vad = vad.cpu().numpy()
    else:
        vad = np.array(vad)

    # Rescale [0, 1] -> [-1, 1]
    vad = vad * 2 - 1
    return vad


# ─────────────────────────────────────────────────────────────
# 2) Convert Cartesian (V, A, D) to spherical coordinates
# ─────────────────────────────────────────────────────────────
def cartesian_to_spherical(vad_arr: np.ndarray) -> np.ndarray:
    """
    Convert VAD vectors from Cartesian coordinates to spherical coordinates.

    Input:
        vad_arr: shape (N, 3) or (3,), interpreted as (V, A, D)

    Output:
        spherical: shape (N, 3) or (3,), interpreted as [r, theta, phi]
          - r: magnitude
          - theta: angle on the V-A plane (azimuth)
          - phi: polar angle relative to D axis (polar)
    """
    v = vad_arr[..., 0]
    a = vad_arr[..., 1]
    d = vad_arr[..., 2]

    r = np.sqrt(v**2 + a**2 + d**2)
    theta = np.arctan2(a, v)              # azimuth in the V-A plane
    phi = np.arccos(d / (r + 1e-8))       # polar angle
    return np.stack([r, theta, phi], axis=-1)


def spherical_angle_dir(sph_words: np.ndarray, sph_nv: np.ndarray) -> np.ndarray:
    """
    Compute the 3D central angle (rotation angle) between direction unit vectors
    defined by spherical coordinates.

    Args:
        sph_words: shape (N, 3) = [r, theta, phi]
        sph_nv:    shape (3,)   = [r, theta, phi] for a single NV

    Returns:
        angles: shape (N,) angles (radians) between each word direction and NV direction.
    """
    theta_words = sph_words[:, 1]
    phi_words = sph_words[:, 2]

    theta_nv = sph_nv[1]
    phi_nv = sph_nv[2]

    # Spherical -> unit vector (x, y, z)
    x_w = np.sin(phi_words) * np.cos(theta_words)
    y_w = np.sin(phi_words) * np.sin(theta_words)
    z_w = np.cos(phi_words)

    x_nv = np.sin(phi_nv) * np.cos(theta_nv)
    y_nv = np.sin(phi_nv) * np.sin(theta_nv)
    z_nv = np.cos(phi_nv)

    dot = x_w * x_nv + y_w * y_nv + z_w * z_nv
    dot = np.clip(dot, -1.0, 1.0)

    return np.arccos(dot)


# ─────────────────────────────────────────────────────────────
# 3) Part A: Build insertion JSON files
# ─────────────────────────────────────────────────────────────
def build_insertion_jsons(
    eecs_dir: str,
    vvs_dir: str,
    nvs_vad_dir: str,
    out_json_dir: str,
) -> None:
    """
    For each EECS utterance file (*.txt), load per-word VAD vectors (VV side),
    load NV VAD vectors, compute angle-based insertion costs, and write results
    to a JSON file per utterance.
    """
    os.makedirs(out_json_dir, exist_ok=True)

    utterance_paths = glob(os.path.join(eecs_dir, "*.txt"))
    for txt_path in tqdm(utterance_paths, desc="Building insertion JSONs"):
        base = os.path.splitext(os.path.basename(txt_path))[0]

        # Parse NV items from the EECS file line: "path,score|path,score|..."
        with open(txt_path, "r", encoding="utf-8") as f:
            line = f.read().strip()

        if not line:
            print(f"[WARN] Empty EECS file: {txt_path}")
            continue

        nv_items = line.split("|")
        # Extract NV wav basename before extension; also strip any ",score" suffix safely
        nv_ids = []
        for item in nv_items:
            path_part = item.split(",")[0].strip()  # keep only the path portion
            nv_ids.append(os.path.splitext(os.path.basename(path_part))[0])

        # Load per-word VAD (.pt only)
        pt_files = glob(os.path.join(vvs_dir, f"{base}_*_*.pt"))
        word_entries = []
        for pt_path in sorted(pt_files):
            name = os.path.basename(pt_path)
            parts = name.split("_")
            try:
                idx = int(parts[-2])                 # word index
                word = parts[-1].replace(".pt", "")  # word token
            except Exception:
                print(f"[WARN] Unexpected VV filename format: {name}")
                continue

            try:
                vad = load_vad(pt_path)  # shape (3,)
            except Exception as e:
                print(f"[ERROR] Failed to load VV VAD: {pt_path} ({e})")
                continue

            word_entries.append((idx, word, vad))

        word_entries.sort(key=lambda x: x[0])
        if not word_entries:
            print(f"[WARN] No per-word VAD .pt files found for utterance: {base}")
            continue

        words = [we[1] for we in word_entries]
        vad_words = np.vstack([we[2] for we in word_entries])  # (N_words, 3)
        N = len(words)
        if N == 0:
            continue

        sph_words = cartesian_to_spherical(vad_words)

        results = {}
        # For each NV, find the best insertion position by minimizing the angle cost
        for nv_id in tqdm(nv_ids, desc=f"{base}: NVs", leave=False):
            pt_nv = os.path.join(nvs_vad_dir, nv_id + ".pt")

            try:
                vad_nv = load_vad(pt_nv)  # (3,)
            except Exception as e:
                print(f"[ERROR] Failed to load NV VAD: {pt_nv} ({e})")
                continue

            sph_nv = cartesian_to_spherical(vad_nv[None, :])[0]
            dists = spherical_angle_dir(sph_words, sph_nv)  # (N,)

            # Compute insertion cost at each position r=0..N
            change_vals = []
            for r in range(N + 1):
                if r == 0:
                    cv = float(dists[0])
                    label = f"before_{words[0]}"
                elif r == N:
                    cv = float(dists[-1])
                    label = f"after_{words[-1]}"
                else:
                    cv = float((dists[r - 1] + dists[r]) / 2.0)
                    label = f"between_{words[r - 1]}_{words[r]}"
                change_vals.append((label, cv))

            best_label, best_val = min(change_vals, key=lambda x: x[1])

            results[nv_id] = {
                "best_insertion": best_label,
                "best_change": float(best_val),
                "all_changes": change_vals,
            }

        out_json = os.path.join(out_json_dir, f"{base}_insertion.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(
                {"utterance": base, "words": words, "nv_ids": nv_ids, "results": results},
                f,
                ensure_ascii=False,
                indent=2,
            )

        print(f"[OK] Wrote insertion JSON: {out_json}")


# ─────────────────────────────────────────────────────────────
# 4) Part B: Convert insertion JSON files to per-NV txt files
# ─────────────────────────────────────────────────────────────
def jsons_to_txts(json_dir: str, out_txt_dir: str) -> None:
    """
    Read *_insertion.json files and write per-NV text files.
    Each output file contains comma-separated change values for all insertion positions.
    """
    os.makedirs(out_txt_dir, exist_ok=True)

    json_paths = glob(os.path.join(json_dir, "*_insertion.json"))
    for json_path in tqdm(json_paths, desc="Converting JSON -> TXT"):
        base = os.path.basename(json_path).replace("_insertion.json", "")

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        results = data.get("results", {})
        if not results:
            print(f"[WARN] No 'results' in JSON: {json_path}")
            continue

        for nv_id, info in results.items():
            out_path = os.path.join(out_txt_dir, f"{base}_{nv_id}.txt")

            all_changes = info.get("all_changes", [])
            changes = [str(change) for _, change in all_changes]

            with open(out_path, "w", encoding="utf-8") as w:
                w.write(",".join(changes))

            print(f"[OK] Saved TXT: {out_path}")


def parse_args():
    p = argparse.ArgumentParser(
        description="End-to-end pipeline: build spherical-angle insertion JSONs and export per-NV TXT files."
    )

    # Input paths
    p.add_argument("--eecs_dir", default="/dataset/EECSscore_top10",
                   help="Directory containing EECS utterance files (*.txt).")
    p.add_argument("--vvs_dir", default="/dataset/VVs_word_vad",
                   help="Directory containing per-word VV VAD vectors (*.pt).")
    p.add_argument("--nvs_vad_dir", default="/dataset/NVs_vad",
                   help="Directory containing NV VAD vectors (*.pt).")

    # Output paths
    p.add_argument("--out_json_dir", default="/dataset/sphere_insertion_results",
                   help="Directory to write insertion JSON results.")
    p.add_argument("--out_txt_dir", default="/dataset/sphere_insertion_txt_results",
                   help="Directory to write per-NV TXT exports.")

    # Execution control
    p.add_argument("--skip_json", action="store_true",
                   help="Skip JSON generation step (Part A).")
    p.add_argument("--skip_txt", action="store_true",
                   help="Skip TXT export step (Part B).")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────
# 5) Main: Run both steps end-to-end
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()

    if not args.skip_json:
        build_insertion_jsons(
            eecs_dir=args.eecs_dir,
            vvs_dir=args.vvs_dir,
            nvs_vad_dir=args.nvs_vad_dir,
            out_json_dir=args.out_json_dir,
        )

    if not args.skip_txt:
        jsons_to_txts(
            json_dir=args.out_json_dir,
            out_txt_dir=args.out_txt_dir,
        )

    print("[DONE] Completed requested steps.")

#!/usr/bin/env python3
import os
from glob import glob
from tqdm import tqdm
import argparse

import torch
import torchaudio
import numpy as np

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks


def extract_emotion2vec_embeddings(wav_dir: str, emb_dir: str, inference_pipeline, skip_existing: bool = False):
    """
    Extract emotion2vec embeddings from all WAV files in wav_dir
    and save them as .pt files in emb_dir, preserving the directory structure.

    Args:
        wav_dir: Input WAV root directory.
        emb_dir: Output embedding root directory.
        inference_pipeline: ModelScope pipeline object.
        skip_existing: If True, skip extraction when the output .pt already exists.
    """
    wav_files = glob(os.path.join(wav_dir, "**", "*.wav"), recursive=True)

    for wav_file in tqdm(wav_files, desc=f"Embedding extraction: {wav_dir}"):
        rel_path = os.path.relpath(wav_file, wav_dir)
        save_path = os.path.splitext(os.path.join(emb_dir, rel_path))[0] + ".pt"

        if skip_existing and os.path.isfile(save_path):
            continue

        # Skip files with zero audio frames
        try:
            info = torchaudio.info(wav_file)
            if info.num_frames == 0:
                print(f"[SKIP] Zero-frame audio file: {wav_file}")
                continue
        except Exception as e:
            print(f"[ERROR] Cannot read audio info: {wav_file} ({e})")
            continue

        # Run emotion2vec inference
        try:
            result = inference_pipeline([wav_file], granularity="utterance")
        except Exception as e:
            print(f"[ERROR] Inference failed: {wav_file} ({e})")
            continue

        # Extract features and save as .pt
        saved_any = False
        for res in result:
            feats = res.get("feats")
            if feats is None:
                print(f"[WARN] No 'feats' found in result for: {wav_file}")
                continue

            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(feats, save_path)
            saved_any = True

        if saved_any:
            print(f"[OK] Saved embedding: {save_path}")


def cosine_similarity(x, y, eps=1e-12):
    """Compute cosine similarity between two vectors (flattened)."""
    x = np.array(x).reshape(-1)
    y = np.array(y).reshape(-1)
    denom = (np.sqrt(np.dot(x, x)) * np.sqrt(np.dot(y, y)) + eps)
    return float(np.dot(x, y) / denom)


def load_tensor_as_numpy(pt_path: str):
    """
    Load a .pt file and return it as a numpy array.
    Supports both torch.Tensor and array-like objects.
    """
    data = torch.load(pt_path, map_location="cpu")
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return np.array(data)


def compute_topk_similar_nvs(
    vvs_emb_dir: str,
    nvs_emb_dir: str,
    output_vv_dir: str,
    nvs_wav_base: str,
    top_k: int = 10,
    vv_name_tags=("sentences", "freeform"),
):
    """
    For each VV embedding (filtered by vv_name_tags), compute cosine similarity
    against NV embeddings from the same speaker, then save top-k NV wav paths
    and similarity scores to a per-VV text file.

    Output format per file:
      path1,score1|path2,score2|...|pathK,scoreK
    """
    os.makedirs(output_vv_dir, exist_ok=True)

    all_vvs = glob(os.path.join(vvs_emb_dir, "**", "*.pt"), recursive=True)
    vvs_ref_pts = [p for p in all_vvs if any(tag in os.path.basename(p) for tag in vv_name_tags)]

    nvs_ref_pts = glob(os.path.join(nvs_emb_dir, "**", "*.pt"), recursive=True)

    # Index NV embeddings by speaker for faster lookup
    nvs_by_speaker = {}
    for p in nvs_ref_pts:
        spk = os.path.basename(os.path.dirname(p))
        nvs_by_speaker.setdefault(spk, []).append(p)

    for v_path in tqdm(vvs_ref_pts, desc="Similarity search (VV -> NV)"):
        speaker = os.path.basename(os.path.dirname(v_path))
        vv_basename = os.path.splitext(os.path.basename(v_path))[0]

        nv_paths_same_spk = nvs_by_speaker.get(speaker, [])
        if not nv_paths_same_spk:
            print(f"[WARN] No NV embeddings found for speaker '{speaker}' (VV: {v_path})")
            continue

        v_emb = load_tensor_as_numpy(v_path)

        sims = []
        for nv_path in nv_paths_same_spk:
            try:
                nv_emb = load_tensor_as_numpy(nv_path)
                score = cosine_similarity(v_emb, nv_emb)
                sims.append((nv_path, score))
            except Exception as e:
                print(f"[ERROR] Failed to load/compare NV: {nv_path} ({e})")
                continue

        if not sims:
            print(f"[WARN] No valid similarity results for VV: {v_path}")
            continue

        topk = sorted(sims, key=lambda x: x[1], reverse=True)[:top_k]

        items = []
        for nv_pt, score in topk:
            nv_base = os.path.splitext(os.path.basename(nv_pt))[0]
            wav_path = os.path.join(nvs_wav_base, f"{speaker}_{nv_base}.wav")
            items.append(f"{wav_path},{score:.6f}")

        out_fn = os.path.join(output_vv_dir, f"{speaker}_{vv_basename}.txt")
        with open(out_fn, "w", encoding="utf-8") as fout:
            fout.write("|".join(items) + "\n")

    print(f"[DONE] Saved top-{top_k} NV (path, similarity) lists for each VV into: {output_vv_dir}")


def parse_args():
    p = argparse.ArgumentParser(
        description="End-to-end pipeline: (1) extract emotion2vec embeddings, (2) compute VV->NV cosine similarity top-k."
    )

    # Model / pipeline options
    p.add_argument("--model", default="iic/emotion2vec_plus_base",
                   help="ModelScope model name for emotion2vec pipeline.")
    p.add_argument("--granularity", default="utterance",
                   help="Granularity passed to the pipeline (default: utterance).")

    # Dataset paths
    p.add_argument("--vv_wav_dir", default="/dataset/EARS_split",
                   help="Root directory of VV WAV files.")
    p.add_argument("--vv_emb_dir", default="/dataset/EARS_split_e2v_emb",
                   help="Root directory to save VV embeddings.")
    p.add_argument("--nv_wav_dir", default="/dataset/EARS_NVs",
                   help="Root directory of NV WAV files.")
    p.add_argument("--nv_emb_dir", default="/dataset/EARS_NVs_e2v_emb",
                   help="Root directory to save NV embeddings.")

    # Similarity output
    p.add_argument("--output_EECS_dir", default="/dataset/EECSscore_top10",
                   help="Output directory for per-VV similarity results.")

    # Behavior options
    p.add_argument("--top_k", type=int, default=10,
                   help="Number of top NV candidates to save for each VV (default: 10).")
    p.add_argument("--vv_name_tags", nargs="+", default=["sentences", "freeform"],
                   help="Only VV embedding files whose basename contains any of these tags are used.")
    p.add_argument("--skip_existing_emb", action="store_true",
                   help="Skip embedding extraction if output .pt already exists.")
    p.add_argument("--skip_embedding", action="store_true",
                   help="Skip embedding extraction step entirely (run similarity only).")
    p.add_argument("--skip_similarity", action="store_true",
                   help="Skip similarity step entirely (run embedding only).")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Initialize pipeline once
    inference_pipeline = pipeline(
        task=Tasks.emotion_recognition,
        model=args.model
    )

    # Step 1) Embedding extraction (VV + NV)
    if not args.skip_embedding:
        extract_emotion2vec_embeddings(
            wav_dir=args.vv_wav_dir,
            emb_dir=args.vv_emb_dir,
            inference_pipeline=inference_pipeline,
            skip_existing=args.skip_existing_emb,
        )
        extract_emotion2vec_embeddings(
            wav_dir=args.nv_wav_dir,
            emb_dir=args.nv_emb_dir,
            inference_pipeline=inference_pipeline,
            skip_existing=args.skip_existing_emb,
        )

    # Step 2) Similarity search
    if not args.skip_similarity:
        compute_topk_similar_nvs(
            vvs_emb_dir=args.vv_emb_dir,
            nvs_emb_dir=args.nv_emb_dir,
            output_vv_dir=args.output_vv_dir,
            nvs_wav_base=args.nv_wav_dir,
            top_k=args.top_k,
            vv_name_tags=tuple(args.vv_name_tags),
        )

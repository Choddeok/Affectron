#!/usr/bin/env python3
import os
import argparse
from glob import glob
from tqdm import tqdm

import torch
import torchaudio
import numpy as np

import torch.nn as nn
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)

# ─────────────────────────────────────────────────────────────
# 1) VAD regression model definition
# ─────────────────────────────────────────────────────────────
class RegressionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        return self.out_proj(x)


class EmotionModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.init_weights()

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs.last_hidden_state
        pooled = hidden_states.mean(dim=1)
        return pooled, self.classifier(pooled)


# ─────────────────────────────────────────────────────────────
# 2) Helpers
# ─────────────────────────────────────────────────────────────
def init_vad_model(model_name: str, device: torch.device):
    """
    Initialize the HuggingFace processor + emotion regression model for VAD extraction.
    """
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = EmotionModel.from_pretrained(model_name).to(device)
    model.eval()
    return processor, model


def extract_vad(wav_path: str, processor, model, device: torch.device) -> np.ndarray:
    """
    Extract a (Valence, Arousal, Dominance) vector from a WAV file.

    Notes:
      - Assumes mono audio. If multi-channel, it will use the first channel.
      - Returns a numpy array of shape (3,).
    """
    wav, sr = torchaudio.load(wav_path)

    # Use the first channel if multi-channel audio
    if wav.ndim == 2 and wav.shape[0] > 1:
        wav = wav[:1, :]

    y = wav.squeeze(0).numpy()
    inputs = processor(y, sampling_rate=sr, return_tensors="pt").input_values.to(device)

    with torch.no_grad():
        _, logits = model(inputs)

    return logits.squeeze(0).detach().cpu().numpy()


def save_vad(vad_vals: np.ndarray, out_dir: str, basename: str, save_pt: bool = True, save_txt: bool = True):
    """
    Save VAD values to .pt and/or .txt files.
    """
    os.makedirs(out_dir, exist_ok=True)

    if save_pt:
        pt_path = os.path.join(out_dir, f"{basename}.pt")
        torch.save(torch.tensor(vad_vals, dtype=torch.float32), pt_path)

    if save_txt:
        txt_path = os.path.join(out_dir, f"{basename}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(" ".join(f"{v:.6f}" for v in vad_vals))


def process_directory(
    wav_glob_pattern: str,
    out_dir: str,
    processor,
    model,
    device: torch.device,
    skip_existing: bool,
    save_pt: bool,
    save_txt: bool,
    desc: str,
):
    """
    Iterate over WAV files matching wav_glob_pattern, extract VAD, and save results.
    """
    wav_files = sorted(glob(wav_glob_pattern))
    if not wav_files:
        print(f"[WARN] No WAV files found: pattern={wav_glob_pattern}")
        return

    for wav_path in tqdm(wav_files, desc=desc):
        basename = os.path.splitext(os.path.basename(wav_path))[0]

        # Skip if outputs already exist
        pt_path = os.path.join(out_dir, f"{basename}.pt")
        txt_path = os.path.join(out_dir, f"{basename}.txt")
        if skip_existing:
            need_pt = (not save_pt) or (os.path.isfile(pt_path))
            need_txt = (not save_txt) or (os.path.isfile(txt_path))
            if need_pt and need_txt:
                continue

        try:
            vad_vals = extract_vad(wav_path, processor, model, device)
            save_vad(vad_vals, out_dir, basename, save_pt=save_pt, save_txt=save_txt)
        except Exception as e:
            print(f"[ERROR] VAD extraction failed: {wav_path} -> {e}")


# ─────────────────────────────────────────────────────────────
# 3) CLI
# ─────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Unified VAD extraction pipeline: extract VAD for VV word-segment WAVs and NV WAVs using a Wav2Vec2-based regression model."
    )

    # Model options
    p.add_argument(
        "--vad_model_name",
        default="audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim",
        help="HuggingFace model name used for VAD regression.",
    )
    p.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device selection. 'auto' uses CUDA if available.",
    )

    # VV (word segments) inputs/outputs
    p.add_argument(
        "--vv_wav_dir",
        default="/dataset/EARS_split",
        help="Directory containing segmented VV word-level WAV files.",
    )
    p.add_argument(
        "--vv_vad_outdir",
        default="/dataset/VVs_word_vad",
        help="Output directory for VV word-level VAD files (.pt/.txt).",
    )
    p.add_argument(
        "--vv_pattern",
        default="*.wav",
        help="Glob pattern for VV segment WAV files inside --vv_seg_wav_dir (default: *.wav).",
    )

    # NV inputs/outputs
    p.add_argument(
        "--nv_wav_dir",
        default="/dataset/EARS_NVs",
        help="Directory containing NV WAV files.",
    )
    p.add_argument(
        "--nv_vad_outdir",
        default="/dataset/NVs_vad",
        help="Output directory for NV VAD files (.pt/.txt).",
    )
    p.add_argument(
        "--nv_pattern",
        default="*.wav",
        help="Glob pattern for NV WAV files inside --nv_wav_dir (default: *.wav).",
    )

    # Behavior flags
    p.add_argument(
        "--run_vv",
        action="store_true",
        help="Run VAD extraction for VV word-segment WAV files.",
    )
    p.add_argument(
        "--run_nv",
        action="store_true",
        help="Run VAD extraction for NV WAV files.",
    )
    p.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip files if the output(s) already exist.",
    )
    p.add_argument(
        "--save_pt",
        action="store_true",
        help="Save outputs as .pt files.",
    )
    p.add_argument(
        "--save_txt",
        action="store_true",
        help="Save outputs as .txt files.",
    )

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Defaults: if user doesn't specify save_pt/save_txt, do both.
    if not args.save_pt and not args.save_txt:
        args.save_pt = True
        args.save_txt = True

    # Defaults: if user doesn't specify run_vv/run_nv, run both.
    if not args.run_vv and not args.run_nv:
        args.run_vv = True
        args.run_nv = True

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Loading VAD model: {args.vad_model_name}")

    processor, model = init_vad_model(args.vad_model_name, device)

    # Run VV processing (segmented word WAVs)
    if args.run_vv:
        vv_glob = os.path.join(args.vv_seg_wav_dir, args.vv_pattern)
        process_directory(
            wav_glob_pattern=vv_glob,
            out_dir=args.vv_vad_outdir,
            processor=processor,
            model=model,
            device=device,
            skip_existing=args.skip_existing,
            save_pt=args.save_pt,
            save_txt=args.save_txt,
            desc="Extracting VAD (VV word segments)",
        )
        print(f"[DONE] VV VAD saved under: {args.vv_vad_outdir}")

    # Run NV processing
    if args.run_nv:
        nv_glob = os.path.join(args.nv_wav_dir, args.nv_pattern)
        process_directory(
            wav_glob_pattern=nv_glob,
            out_dir=args.nv_vad_outdir,
            processor=processor,
            model=model,
            device=device,
            skip_existing=args.skip_existing,
            save_pt=args.save_pt,
            save_txt=args.save_txt,
            desc="Extracting VAD (NVs)",
        )
        print(f"[DONE] NV VAD saved under: {args.nv_vad_outdir}")

    print("[DONE] Completed VAD extraction pipeline.")

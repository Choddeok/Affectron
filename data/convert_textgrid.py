#!/usr/bin/env python3
import os
from pathlib import Path
from praatio import tgio

def extract_frames_only(input_dir: Path,
                        output_dir: Path,
                        frame_rate: float = 50.0):
    """
    For all .TextGrid files in input_dir,
    extract only the start_frame of each word and the final end_frame
    from the word tier and save them as `{base_name}.frames.txt`.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for tg_path in sorted(input_dir.glob("*.TextGrid")):
        try:
            tg = tgio.openTextgrid(str(tg_path))
            word_tier = tg.tierDict.get("words")
            if word_tier is None:
                print(f"[WARN] 'words' tier not found: {tg_path.name}")
                continue

            # Remove empty labels
            raw = [(s, e, lbl) for s, e, lbl in word_tier.entryList if lbl.strip()]
            if not raw:
                print(f"[WARN] No words found: {tg_path.name}")
                continue

            # Extract frames only
            frames = []
            for idx, (start, end, _) in enumerate(raw):
                start_frame = round(start * frame_rate)
                frames.append(start_frame)
                
            # Add end_frame of the last word
            last_end_frame = round(raw[-1][1] * frame_rate)
            frames.append(last_end_frame)

            # Save
            out_path = output_dir / f"{tg_path.stem}.frames.txt"
            with open(out_path, "w", encoding="utf-8") as wf:
                for frm in frames:
                    wf.write(f"{frm}\n")

            print(f"[OK] {tg_path.name} → {out_path.name}")

        except Exception as e:
            print(f"[ERROR] 처리 실패 {tg_path.name}: {e}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input_dir", required=True,
                   help="Path to the TextGrid directory")
    p.add_argument("-o", "--output_dir", required=True,
                   help="Path to the output directory")
    p.add_argument("-f", "--frame_rate", type=float, default=50.0,
                   help="Seconds-to-frames ratio (default: 50)")
    args = p.parse_args()
    
    extract_frames_only(Path(args.input_dir),
                        Path(args.output_dir),
                        frame_rate=args.frame_rate)

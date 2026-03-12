#!/usr/bin/env python3
import argparse
from pathlib import Path
import torch
import torchaudio
from data.tokenizer import AudioTokenizer

def parse_args():
    p = argparse.ArgumentParser(
        description="Decode saved codebook tokens (y.txt) back into audio via EnCodec"
    )
    p.add_argument(
        "--y_dir",
        type=Path,
        help="Directory containing *_y.txt files you saved earlier",
        default="./test"
    )
    p.add_argument(
        "--signature",
        type=str,
        help="path to the EnCodec model checkpoint (same as in training)",
        default="/checkpoints/encodec/encodec_4cb2048_giga.th"
    )
    p.add_argument(
        "--output_dir",
        type=Path,
        help="where to write the .wav files",
        default="./test"
    )
    p.add_argument(
        "--codec_audio_sr",
        type=int,
        default=16000,
        help="sample rate for the decoded wave",
    )
    return p.parse_args()

def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1) load the codec (AudioTokenizer 내부의 .codec 에 모델이 올라갑니다)
    audio_tokenizer = AudioTokenizer(signature=args.signature)
    # 내부 codec의 디바이스를 꺼내옵니다
    codec = audio_tokenizer.codec
    codec_device = next(codec.parameters()).device

    # 2) glob all *_y.txt files
    for y_path in sorted(args.y_dir.glob("*_y.txt")):
        # read each line as one codebook's token sequence
        lines = y_path.read_text().splitlines()
        codebooks = [list(map(int, L.split())) for L in lines]
        # stack into a tensor of shape [1, K, T]
        frames = torch.LongTensor(codebooks).unsqueeze(0)  # [1, K, T]
        # 모델이 있는 디바이스로 이동
        frames = frames.to(codec_device)

        # decode
        wav = audio_tokenizer.decode([(frames, None)])  # 이제 frames와 codec이 같은 디바이스
        # 반환 리스트의 첫 번째 요소가 Tensor
        wav = wav[0].cpu()  # 텐서를 CPU로 꺼내서 저장 준비

        # save
        out_name = y_path.stem + ".wav"
        out_path = args.output_dir / out_name
        torchaudio.save(out_path, wav, args.codec_audio_sr)
        print(f"Decoded {y_path.name} → {out_name}")

if __name__ == "__main__":
    main()

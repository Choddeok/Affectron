input_fn  = "final_test_meta_unseen.txt"
output_fn = "final_test_meta_unseen2.txt"

seen = set()

with open(input_fn, "r", encoding="utf-8") as fin, \
     open(output_fn, "w", encoding="utf-8") as fout:

    for line in fin:
        line = line.strip()
        if not line:
            continue
        
        parts = line.split("|", 2)  # 0: wav_meta, 1: new_wav, 2: merged text
        if len(parts) < 3:
            continue
        
        wav_meta = parts[0]  # ★ 중복 판단 기준이 되는 부분

        # 이미 등장한 경우 → 스킵
        if wav_meta in seen:
            continue
        
        # 처음 등장한 경우 → 저장
        seen.add(wav_meta)
        fout.write(line + "\n")

print(f"[DONE] 중복 제거 완료. {output_fn} 에 저장했습니다.")

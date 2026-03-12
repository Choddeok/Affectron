# Convert TextGrid alignments to frame-level text files
python convert_textgrid.py \
  --input_dir /dataset/TextGrid_mfa \
  --output_dir /dataset/TextGrid_txt \
  --frame_rate 50

# Extract emotion2vec embeddings (utterance-level)
python emotion2vec_extract.py \
    --vv_wav_dir /dataset/EARS_split \
    --vv_emb_dir /dataset/EARS_split_e2v_emb \
    --nv_wav_dir /dataset/EARS_NVs \
    --nv_emb_dir /dataset/EARS_NVs_e2v_emb \
    --output_EECS_dir /dataset/EECSscore_top10 \
    --top_k 10 

# Extract emotion attributes (Valence–Arousal–Dominance) 
python emotion_attributes_extract.py \
    --vv_wav_dir /dataset/EARS_split \
    --vv_vad_outdir /dataset/VVs_word_vad \
    --nv_wav_dir /dataset/EARS_NVs \
    --nv_vad_outdir /dataset/NVs_vad 

# Estimate NV insertion positions in spherical space
python spherical_insertion_pipeline.py \
    --eecs_dir /dataset/EECSscore_top10 \
    --vvs_dir /dataset/VVs_word_vad \
    --nvs_vad_dir /dataset/NVs_vad \
    --out_json_dir /dataset/sphere_insertion_results \
    --out_txt_dir /dataset/sphere_insertion_txt_results 

# Encodec encoding and phoneme extraction
python phonemize_encodec_encode_hf.py \
    --input_dir /dataset/EARS_split \
    --meta ./meta.txt \
    --save_dir /dataset/EARS_TTS_encodec \
    --encodec_model_path /checkpoints/encodec/encodec_4cb2048_giga.th \
    --mega_batch_size 120 \
    --batch_size 32 \
    --max_len 30000


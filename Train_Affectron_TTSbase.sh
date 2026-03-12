export CUDA_VISIBLE_DEVICES=0,1,2,3
export WORLD_SIZE=4

dataset=ears
model_name=Affectron_TTSbase
mkdir -p /out/affectron/${model_name}/logs/${dataset}
exp_root="/out/affectron/${model_name}"
exp_name=e330M_ft
dataset_dir="/dataset/EARS_TTS_encodec"
encodec_codes_folder_name="encodec_16khz_4codebooks"
load_model_from="./pretrained_models/330M_TTSEnhanced.pth"

#####################
#       Train       #
#####################
torchrun --nnodes=1 --nproc_per_node=${WORLD_SIZE} \
main.py \
--load_model_from ${load_model_from} \
--reduced_eog 1 \
--drop_long 1 \
--eos 2051 \
--n_special 4 \
--pad_x 0 \
--codebook_weight "[2,1,1,1]" \
--encodec_sr 50 \
--num_steps 50000 \
--lr 1e-05 \
--batch_size 100 \
--warmup_fraction 0.1 \
--optimizer_name "AdamW" \
--d_model 1024 \
--audio_embedding_dim 1024 \
--nhead 16 \
--num_decoder_layers 24 \
--max_num_tokens 20000 \
--gradient_accumulation_steps 24 \
--val_max_num_tokens 6000 \
--num_buckets 10 \
--audio_max_length 16.0 \
--audio_min_length 1.0 \
--text_max_length 400 \
--text_min_length 10.0 \
--mask_len_min 1 \
--mask_len_max 600 \
--tb_write_every_n_steps 100 \
--print_every_n_steps 400 \
--val_every_n_steps 10000 \
--text_vocab_size 130 \
--text_pad_token 130 \
--phn_folder_name "phonemes" \
--manifest_name "manifest" \
--encodec_folder_name ${encodec_codes_folder_name} \
--audio_vocab_size 2048 \
--empty_token 2048 \
--eog 2049 \
--audio_pad_token 2050 \
--n_codebooks 4 \
--max_n_spans 3 \
--shuffle_mask_embedding 0 \
--mask_sample_dist poisson1 \
--max_mask_portion 0.9 \
--min_gap 5 \
--num_workers 8 \
--dynamic_batching 1 \
--dataset $dataset \
--exp_dir "${exp_root}/${dataset}/${exp_name}" \
--dataset_dir ${dataset_dir} \
--data_augmentation 1 \
--emotion_driven_nv_matching  1 \
--emotion_aware_routing  1 \
--min_based_position_select 0 \
--emotion_aware_structural_masking 1 \
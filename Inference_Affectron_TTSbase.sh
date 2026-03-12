# ##################
#       test       #
####################
export CUDA_VISIBLE_DEVICES=1
exp_dir="/out/affectron/${model_name}/ears/e330M_ft"
VV_seen_output_dir="/out/affectron/${model_name}/wavs/seen_VV"
VVNV_seen_output_dir="/out/affectron/${model_name}/wavs/seen_VVNV"
VV_unseen_output_dir="/out/affectron/${model_name}/wavs/unseen_VV"
VVNV_unseen_output_dir="/out/affectron/${model_name}/wavs/unseen_VVNV"

# VV_seen
for step in 50000; do
  python inference_tts.py \
    --exp_dir "$exp_dir" \
    --ckpt_fn "bundle_step_${step}.pth" \
    --output_dir "$VV_seen_output_dir" \
    --manifest_fn "./manifest/test_VV_seen.txt"
done

# VVNV_seen
for step in 50000; do
  python inference_tts.py \
    --exp_dir "$exp_dir" \
    --ckpt_fn "bundle_step_${step}.pth" \
    --output_dir "$VVNV_seen_output_dir" \
    --manifest_fn "./manifest/test_VVNV_seen.txt"
done

# VV_unseen
for step in 50000; do
  python inference_tts.py \
    --exp_dir "$exp_dir" \
    --ckpt_fn "bundle_step_${step}.pth" \
    --output_dir "$VV_unseen_output_dir" \
    --manifest_fn "./manifest/test_VV_unseen.txt"
done

# VVNV_unseen
for step in 50000; do
  python inference_tts.py \
    --exp_dir "$exp_dir" \
    --ckpt_fn "bundle_step_${step}.pth" \
    --output_dir "$VVNV_unseen_output_dir" \
    --manifest_fn "./manifest/test_VVNV_unseen.txt"
done
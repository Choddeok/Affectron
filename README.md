<div align="center">

## Affectron: Emotional Speech Synthesis with Affective and Contextually Aligned Nonverbal Vocalizations

</div>
<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="Figures/logo.gif">
  <img src="Figures/logo.gif" alt="Affectron Logo" width="800">
</picture>

</div>

<div align="center">

<a href="https://choddeok.github.io/Affectron/" target="_blank">
  <img src="https://img.shields.io/badge/🎧%20Demo-Listen%20Here-8A2BE2?style=for-the-badge" />
</a>

</div>

<br>

## 📰 News
- **2025-12-21**: We officially released **Affectron**, along with an interactive demo page showcasing affective and contextually aligned nonverbal vocalizations in emotional speech synthesis.
- **2025-12-30**: Added comprehensive **Training** and **Inference** guidance to facilitate reproducibility and ease of use.

<br>

## ⭐ TODO
- [x] Codebase upload
- [x] Environment setup
- [x] Training guidance
- [x] Inference guidance
- [ ] Pretrained checkpoints

<br>

## Introduction
<div align="center">
  <img src="https://github.com/user-attachments/assets/3a81d946-0645-4a46-921a-583b06304bbb" width="75%">
  <br>
  <em>Overall framework of Affectron.</em>
</div>

<br>

Emotional speech synthesis benefits greatly from nonverbal vocalizations (NVs), such as laughter and sighs, which convey affect beyond words. However, NVs are often underrepresented due to limited data availability and reliance on proprietary resources or NV detectors.

We propose **Affectron**, a framework that generates affectively and contextually aligned NVs using NV-augmented training on a small-scale open and decoupled corpus. 

<br>

---

## 0. Environment setup
```bash
conda create -n affectron python=3.9.16
conda activate affectron

pip install -e git+https://github.com/facebookresearch/audiocraft.git@c5157b5bf14bf83449c17ea1eeb66c19fb4bc7f0#egg=audiocraft
pip install xformers==0.0.22
pip install torchaudio==2.0.2 torch==2.0.1 # this assumes your system is compatible with CUDA 11.7, otherwise checkout https://pytorch.org/get-started/previous-versions/#v201
apt-get install ffmpeg # if you don't already have ffmpeg installed
apt-get install espeak-ng # backend for the phonemizer installed below
pip install tensorboard==2.16.2
pip install phonemizer==3.2.1
pip install datasets==2.16.0
pip install torchmetrics==0.11.1
pip install huggingface_hub==0.22.2
# install MFA for getting forced-alignment, this could take a few minutes
conda install -c conda-forge montreal-forced-aligner=2.2.17 openfst=1.8.2 kaldi=5.5.1068
# install MFA english dictionary and model
mfa model download dictionary english_us_arpa
mfa model download acoustic english_us_arpa
```

<br>

---

## 1. Training

This section describes the full training pipeline for Affectron, including dataset preparation, feature extraction, and model training.

<br>

### Step 1) Download EARS dataset and split Verbal / NV recordings

Download the EARS dataset following the official instructions: 
- https://github.com/facebookresearch/ears_dataset

After downloading, split the recordings into **verbal** and **nonverbal vocalization (NV)** subsets as required by the training pipeline.

<br>

### Step 2) Processing

We provide a preprocessing script that:
- loads utterances and their transcripts,
- encodes utterances into discrete codes using **Encodec**,
- converts transcripts into **phoneme sequences**,
- and builds a phoneme vocabulary (`vocab.txt`).

Run the following command: \
✅ Option 1: Run the full pipeline with a single script (recommended) \
This option executes all processing steps sequentially using a shell script.
```bash
conda activate affectron
export CUDA_VISIBLE_DEVICES=0
cd ./data
sh Processing.sh
```

<br>

✅ Option 2: Run each step manually \
Alternatively, you can run each processing step individually as shown below.

```bash
conda activate affectron
export CUDA_VISIBLE_DEVICES=0
cd ./data

# Convert TextGrid alignments to frame-level text files
python convert_textgrid.py \
  --input_dir path/to/TextGrid_mfa \
  --output_dir path/to/TextGrid_txt \
  --frame_rate 50

# Extract emotion2vec embeddings (utterance-level)
python emotion2vec_extract.py \
  --vv_wav_dir path/to/EARS_split \
  --vv_emb_dir path/to/EARS_split_e2v_emb \
  --nv_wav_dir path/to/EARS_NVs \
  --nv_emb_dir path/to/EARS_NVs_e2v_emb \
  --output_EECS_dir path/to/EECSscore_top10 \
  --top_k 10

# Extract emotion attributes (Valence–Arousal–Dominance)
python emotion_attributes_extract.py \
  --vv_wav_dir path/to/EARS_split \
  --vv_vad_outdir path/to/VVs_word_vad \
  --nv_wav_dir path/to/EARS_NVs \
  --nv_vad_outdir path/to/NVs_vad

# Estimate NV insertion positions in spherical VAD space
python spherical_insertion_pipeline.py \
  --eecs_dir path/to/EECSscore_top10 \
  --vvs_dir path/to/VVs_word_vad \
  --nvs_vad_dir path/to/NVs_vad \
  --out_json_dir path/to/sphere_insertion_results \
  --out_txt_dir path/to/sphere_insertion_txt_results

# Encodec encoding and phoneme extraction
python phonemize_encodec_encode_hf.py \
  --input_dir path/to/EARS_split \
  --meta path/to/meta.txt \
  --save_dir path/to/EARS_TTS_encodec \
  --encodec_model_path path/to/encodec_model \
  --mega_batch_size 120 \
  --batch_size 32 \
  --max_len 30000
```

#### Encodec model
Use the **same Encodec model as the VoiceCraft baseline**:
- https://huggingface.co/pyp1/VoiceCraft \
This model is trained on **GigaSpeech XL**, has **56M parameters**, and uses **4 codebooks**, each with **2048 codes**.

#### Emotion2vec model
We use the **Emotion2Vec model** for utterance-level emotional representation and similarity computation.
- https://github.com/ddlBoJack/emotion2vec \
Emotion2Vec produces continuous emotion embeddings that capture high-level affective characteristics of speech. \
In this pipeline, Emotion2Vec embeddings are used to compute EECS-based similarity between VV and NV utterances.

#### Emotion Attributes Model
We use a **wav2vec2-based regression model** to predict continuous **Valence–Arousal–Dominance (VAD)** attributes from speech. \
- https://huggingface.co/audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim \
This model is trained on the **MSP-Podcast emotion dataset** and predicts a 3-dimensional continuous VAD vector for each utterance or word-level segment. \
In our pipeline, these VAD vectors are further transformed into spherical coordinates, and **angular distance in VAD space** is used to estimate optimal NV insertion positions.


<br>

### Step 3) Model training
Start training with:
```bash
sh Train_Affectron_TTSbase.sh
```

Before running the script, make sure to configure the following variables according to your environment:
- `dataset`
- `model_name`
- `exp_name`
- `exp_root`
- `dataset_dir`
- `load_model_from`
Training logs and checkpoints will be saved under `exp_root`.

<br>

---

## 2. Inferece

### Step 1) Create a manifest file

Create a meta file under the `./manifest` directory. \
Each line is **tab-separated** and consists of:
1. Reference audio path
2. Text (reference transcript + target generation text)
3. Utterance ID
4. Prompt audio length (seconds)
5. Prompt start time (seconds)

Example:
```bash
/dataset/EARS_final/VVNVs/p005_emo_adoration_sentences_0.wav	You're just the sweetest person I know, and I'm so happy to call you my friend. I had the best time with you.	p005_emo_adoration_sentences_0	5.9	0.0
```
⚠️ Important (same as the VoiceCraft baseline): \
Ensure **(prompt length + generation length) ≤ 16 seconds**. \
Due to limited compute, utterances longer than **16 seconds** were excluded during training.

<br>

### Step 2) Run inference
```bash
sh Inference_Affectron_TTSbase.sh
```
Before running, set:
- `model_name`
- `exp_dir`
- `output_dir` 
Generated audio files will be saved under `output_dir`.

<br>

--- 
## 3. Pretrained checkpoints

To preserve anonymity and avoid download tracking during the review process, pretrained checkpoints will be released **after the review is completed**.

<br>

---
## 4. Acknowledgements
**Our codes are based on the following repos:**
* [VoiceCraft](https://github.com/jasonppy/VoiceCraft/tree/master)

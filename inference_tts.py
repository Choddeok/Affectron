import argparse, pickle
import logging
import os, random
import numpy as np
import torch
import torchaudio
import re

from data.tokenizer import (
    AudioTokenizer,
    TextTokenizer,
    tokenize_audio,
    tokenize_text
)

from models import voicecraft
import argparse, time, tqdm


# this script only works for the musicgen architecture
def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--manifest_fn", type=str, default="./manifest/final_test_manifest.txt")
    parser.add_argument("--audio_root", type=str, default="/dataset/EARS_final/VVs")
    parser.add_argument("--exp_dir", type=str, default="/out/voicecraft/gigaspeech/giga/e830M")
    parser.add_argument("--ckpt_fn", type=str, default="bundle_step_11200.pth")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--codec_audio_sr", type=int, default=16000, help='the sample rate of audio that the codec is trained for')
    parser.add_argument("--codec_sr", type=int, default=50, help='the sample rate of the codec codes')
    parser.add_argument("--top_k", type=int, default=40, help="sampling param")
    parser.add_argument("--top_p", type=float, default=1, help="sampling param")
    parser.add_argument("--temperature", type=float, default=1.0, help="sampling param")
    parser.add_argument("--output_dir", type=str, default="/out/voicecraft/gigaspeech/giga/wavs")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--signature", type=str, default="/checkpoints/encodec/encodec_4cb2048_giga.th", help="path to the encodec model")
    parser.add_argument("--crop_concat", type=int, default=0)
    parser.add_argument("--stop_repetition", type=int, default=-1, help="used for inference, when the number of consecutive repetition of a token is bigger than this, stop it")
    parser.add_argument("--kvcache", type=int, default=1, help='if true, use kv cache, which is 4-8x faster than without')
    parser.add_argument("--sample_batch_size", type=int, default=1, help="batch size for sampling, NOTE that it's not running inference for several samples, but duplicate one input sample batch_size times, and during inference, we only return the shortest generation")
    parser.add_argument("--silence_tokens", type=str, default="[1388,1898,131]", help="note that if you are not using the pretrained encodec 6f79c6a8, make sure you specified it yourself, rather than using the default")
    return parser.parse_args()


@torch.no_grad()
def inference_one_sample(model, model_args, phn2num, text_tokenizer, audio_tokenizer, audio_fn, target_text, device, decode_config, prompt_lenght_frame, prompt_end_frame):
    parts = re.split(r'(<[^>]+>)', target_text.strip())
    phns = []
    for idx, part in enumerate(parts):
        if not part.strip():
            continue
        if re.fullmatch(r'<[^>]+>', part):
            phns.append("_")
            phns.append(part)
            phns.append("_")
        else:
            phns.extend(tokenize_text(text_tokenizer, part))
    if phns and phns[0] == "_":
        phns.pop(0)
    if phns and phns[-1] == "_":
        phns.pop()
        
    # phonemize
    text_tokens = [phn2num[p] for p in phns if p in phn2num]
    text_tokens = torch.LongTensor(text_tokens).unsqueeze(0)
    text_tokens_lens = torch.LongTensor([text_tokens.shape[-1]])
    # encode audio
    encoded_frames = tokenize_audio(audio_tokenizer, audio_fn, offset=prompt_end_frame, num_frames=prompt_lenght_frame)
    original_audio = encoded_frames[0][0].transpose(2,1) # [1,T,K]
    assert original_audio.ndim==3 and original_audio.shape[0] == 1 and original_audio.shape[2] == model_args.n_codebooks, original_audio.shape
    logging.info(f"original audio length: {original_audio.shape[1]} codec frames, which is {original_audio.shape[1]/decode_config['codec_sr']:.2f} sec.")

    # forward
    stime = time.time()
    if decode_config['sample_batch_size'] <= 1:
        logging.info(f"running inference with batch size 1")
        concat_frames, gen_frames = model.inference_tts(
            text_tokens.to(device),
            text_tokens_lens.to(device),
            original_audio[...,:model_args.n_codebooks].to(device), # [1,T,8]
            top_k=decode_config['top_k'],
            top_p=decode_config['top_p'],
            temperature=decode_config['temperature'],
            stop_repetition=decode_config['stop_repetition'],
            kvcache=decode_config['kvcache'],
            silence_tokens=eval(decode_config['silence_tokens']) if type(decode_config['silence_tokens'])==str else decode_config['silence_tokens']
        ) # output is [1,K,T]
    else:
        logging.info(f"running inference with batch size {decode_config['sample_batch_size']}, i.e. return the shortest among {decode_config['sample_batch_size']} generations.")
        concat_frames, gen_frames = model.inference_tts_batch(
            text_tokens.to(device),
            text_tokens_lens.to(device),
            original_audio[...,:model_args.n_codebooks].to(device), # [1,T,8]
            top_k=decode_config['top_k'],
            top_p=decode_config['top_p'],
            temperature=decode_config['temperature'],
            stop_repetition=decode_config['stop_repetition'],
            kvcache=decode_config['kvcache'],
            batch_size = decode_config['sample_batch_size'],
            silence_tokens=eval(decode_config['silence_tokens']) if type(decode_config['silence_tokens'])==str else decode_config['silence_tokens']
        ) # output is [1,K,T]
    logging.info(f"inference on one sample take: {time.time() - stime:.4f} sec.")

    logging.info(f"generated encoded_frames.shape: {gen_frames.shape}, which is {gen_frames.shape[-1]/decode_config['codec_sr']} sec.")
    
    # for timestamp, codes in enumerate(gen_frames[0].transpose(1,0)):
    #     logging.info(f"{timestamp}: {codes.tolist()}")
    # decode (both original and generated)
    concat_sample = audio_tokenizer.decode(
        [(concat_frames, None)] # [1,T,8] -> [1,8,T]
    )
    gen_sample = audio_tokenizer.decode(
        [(gen_frames, None)]
    )
    #Empty cuda cache between runs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # return
    return concat_sample, gen_sample

def get_model(exp_dir, ckpt, device=None):
    with open(os.path.join(exp_dir, "args.pkl"), "rb") as f:
        model_args = pickle.load(f)

    logging.info("load model weights...")
    model = voicecraft.VoiceCraft(model_args)
    ckpt_fn = os.path.join(exp_dir, ckpt)
    ckpt = torch.load(ckpt_fn, map_location='cpu')['model']
    phn2num = torch.load(ckpt_fn, map_location='cpu')['phn2num']
    model.load_state_dict(ckpt)
    del ckpt
    logging.info("done loading weights...")
    if device == None:
        device = torch.device("cpu")
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
    model.to(device)
    model.eval()
    return model, model_args, phn2num

if __name__ == "__main__":
    def seed_everything(seed):
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    formatter = (
        "%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d || %(message)s"
    )
    logging.basicConfig(format=formatter, level=logging.INFO)
    args = get_args()
    # args.device='cpu'
    seed_everything(args.seed)
    
    ckpt = args.ckpt_fn.split(".")[0]
    
    os.makedirs(os.path.join(args.output_dir, ckpt, "gen"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, ckpt, "concat"), exist_ok=True)
    # load model
    with open(args.manifest_fn, "r") as rf:
        manifest = [l.strip().split("\t") for l in rf.readlines()]
    manifest = manifest[1:]
    manifest = [[item[0], item[1], item[2], item[3], item[4]] for item in manifest]
    
    stime = time.time()
    logging.info(f"loading model from {args.exp_dir}")
    model, model_args, phn2num = get_model(args.exp_dir, args.ckpt_fn)
    logging.info(f"loading model done, took {time.time() - stime:.4f} sec")

    # setup text and audio tokenizer
    text_tokenizer = TextTokenizer(backend="espeak")
    audio_tokenizer = AudioTokenizer(signature=args.signature) # will also put the neural codec model on gpu
    
    audio_fns = []
    texts = []
    prompt_lenght_frames = []
    prompt_end_frames = []
    new_audio_fns = []

    for item in manifest:
        audio_fn = os.path.join(args.audio_root, item[0])
        audio_fns.append(audio_fn)
        temp = torchaudio.info(audio_fn)
        prompt_lenght_frames.append(round(float(item[3])*temp.sample_rate))
        prompt_end_frames.append(round(float(item[4])*temp.sample_rate))
        texts.append(item[1])
        new_audio_fns.append(item[2])

    for i, (audio_fn, text, prompt_end_frame, prompt_lenght_frame, new_audio_fn) in enumerate(tqdm.tqdm((zip(audio_fns, texts, prompt_end_frames, prompt_lenght_frames, new_audio_fns)))):
        new_audio_fn = new_audio_fn.replace("_0", "_1")
        output_expected_sr = args.codec_audio_sr
        seg_save_fn_gen = f"{args.output_dir}/{ckpt}/gen/{new_audio_fn}.wav"
        seg_save_fn_concat = f"{args.output_dir}/{ckpt}/concat/{new_audio_fn}.wav"
        
        if os.path.exists(seg_save_fn_gen) and os.path.exists(seg_save_fn_concat):
            print(f"Skipping {new_audio_fn}, already exists.")
            continue
        
        try:
            concated_audio, gen_audio = inference_one_sample(model, model_args, phn2num, text_tokenizer, audio_tokenizer, audio_fn, text, args.device, vars(args), prompt_lenght_frame, prompt_end_frame)
        except Exception as e:
            continue  
    
        # save segments for comparison
        concated_audio, gen_audio = concated_audio[0].cpu(), gen_audio[0].cpu()
        if output_expected_sr != args.codec_audio_sr:
            gen_audio = torchaudio.transforms.Resample(output_expected_sr, args.codec_audio_sr)(gen_audio)
            concated_audio = torchaudio.transforms.Resample(output_expected_sr, args.codec_audio_sr)(concated_audio)

        torchaudio.save(seg_save_fn_gen, gen_audio, args.codec_audio_sr)
        torchaudio.save(seg_save_fn_concat, concated_audio, args.codec_audio_sr)

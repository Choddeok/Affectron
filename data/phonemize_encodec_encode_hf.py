import argparse
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torchaudio
from tokenizer import TextTokenizer, tokenize_text

class EarsDataset(Dataset):
    def __init__(self, input_dir: Path, meta: Path):
        self.input_dir = input_dir
        self.entries = []
        for L in meta.read_text(encoding="utf-8").splitlines():
            wav_name, txt = L.split("|", 1)
            self.entries.append((wav_name, txt))

    def __len__(self): 
        return len(self.entries)

    def __getitem__(self, idx):
        seg, txt = self.entries[idx]
        wav_path = self.input_dir/seg
        wav, sr = torchaudio.load(wav_path)
        wav = wav.squeeze(0)
        duration = wav.shape[-1]/sr
        return seg, wav, sr, txt, 0.0, duration

    def collate(self, batch):
        batch = [b for b in batch if b[1].numel()>0]
        res = {k:[] for k in ["segment_id","audio","sr","text","begin_time","end_time"]}
        for seg,w,sr,txt,bt,et in batch:
            res["segment_id"].append(seg)
            res["audio"].append(w)
            res["sr"].append(sr)
            res["text"].append(txt)
            res["begin_time"].append(bt)
            res["end_time"].append(et)
        return res
    
def parse_args():
    parser = argparse.ArgumentParser(description="encode the librilight dataset using encodec model")
    parser.add_argument("--input_dir", type=Path, default="/dataset/EARS_split")
    parser.add_argument("--meta", type=Path, default="./meta.txt")
    parser.add_argument('--save_dir', type=str, default="/dataset/EARS_TTS_encodec")
    parser.add_argument('--encodec_model_path', type=str, default="/checkpoints/encodec/encodec_4cb2048_giga.th")
    parser.add_argument('--n_workers', type=int, default=4, help="Number of parallel worker processes")
    parser.add_argument('--mega_batch_size', type=int, default=1, help="Number of samples in each mega batch for multiprocess dataloading")
    parser.add_argument('--batch_size', type=int, default=1, help="batch size for encodec encoding, decrease it if OOM. This is the sum of batch size *over each gpu*, so increase it if you are using more gpus")
    parser.add_argument('--model_sr', type=int, default=16000, help='encodec input audio sample rate')
    parser.add_argument('--downsample_rate', type=int, default=320, help='encodec downsample rate')
    parser.add_argument('--model_code_sr', type=int, default=50, help='encodec model code sample rate')
    parser.add_argument('--len_cap', type=float, default=35.0, help='will drop audios that are longer than this number')
    parser.add_argument('--max_len', type=int, default=30000, help='max length of audio in samples, if exceed, will cut a batch into half to process, decrease this number if OOM on your machine')
    return parser.parse_args()
if __name__ == "__main__":
    import logging
    formatter = (
        "%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d || %(message)s"
    )
    logging.basicConfig(format=formatter, level=logging.INFO)
    args = parse_args()

    import os
    import numpy as np
    import torch
    import tqdm
    import time

    # get the path
    phn_save_root = os.path.join(args.save_dir, "phonemes")
    codes_save_root = os.path.join(args.save_dir, "encodec_16khz_4codebooks")
    vocab_fn = os.path.join(args.save_dir, "vocab.txt")
    os.makedirs(phn_save_root, exist_ok=True)
    os.makedirs(codes_save_root, exist_ok=True)


    def sort_by_audio_len(lens):
        inds = np.argsort(lens).tolist()
        logging.info(f"longest: {lens[inds[-1]]*args.model_code_sr} encodec codes, {lens[inds[-1]]:.2f} sec.")
        logging.info(f"shortest: {lens[inds[0]]*args.model_code_sr} encodec codes, {lens[inds[0]]:.2f} sec.")
        logging.info(f"median: {lens[inds[len(inds)//2]]*args.model_code_sr} encodec codes, {lens[inds[len(inds)//2]]:.2f} sec.")
        logging.info(f"95 percentile longest: {lens[inds[int(len(inds)*0.95)]]*args.model_code_sr} encodec codes, {lens[inds[int(len(inds)*0.95)]]:.2f} sec.")
        return inds[::-1]
    
    def write_array_to_txt_file(array, filename):
        with open(filename, 'w') as f:
            for a in array[:-1]:
                f.write(' '.join(map(str, a))+'\n')
            f.write(' '.join(map(str, array[-1])))
    

    ### phonemization
    # load tokenizer
    # load the encodec model
    from audiocraft.solvers import CompressionSolver
    model = CompressionSolver.model_from_checkpoint(args.encodec_model_path)
    model = model.cuda()
    model = model.eval()
    text_tokenizer = TextTokenizer()


    # https://github.com/SpeechColab/GigaSpeech
    # there are only four different punctuations
    # need to check whether there are other < started strings
    punc2sym = {" <COMMA>": ",", " <PERIOD>": ".", " <QUESTIONMARK>": "?", " <EXCLAMATIONPOINT>": "!"} # note the space in front of each punc name
    gar2sym = {"<SIL>": "#%#", "<MUSIC>": "##%", "<NOISE>": "%%#", "<OTHER>":"%#%","<agreement>":"@@0","<anger>":"@@1","<cheering>":"@@2","<congratulations>":"@@3","<coughing>":"@@4","<crying>":"@@5","<eating>":"@@6","<filler>":"@@7","<greetings>":"@@8","<laughter>":"@@9","<screaming>":"@@10","<sneezing>":"@@11","<throat>":"@@12","<yawning>":"@@13","<yelling>":"@@14"} # so that they are savely keep as the original sym when using tokenize_text
    punc2sym.update(gar2sym)

    word2sym = { "h æ ʃ h ɐ ʃ p ɚ s ɛ n t": "<MUSIC>", 
                "h æ ʃ p ɚ s ɛ n t h æ ʃ": "<SIL>", 
                "p ɚ s ɛ n t h ɐ ʃ p ɚ s ɛ n t": "<OTHER>", 
                "p ɚ s ɛ n t p ɚ s ɛ n t h æ ʃ": "<NOISE>",
                "æ t æ t _ z iə ɹ oʊ":"<agreement>",
                "æ t æ t _ w ʌ n":"<anger>",
                "æ t æ t _ t uː":"<cheering>",
                "æ t æ t _ θ ɹ iː":"<congratulations>",
                "æ t æ t _ f oːɹ":"<coughing>",
                "æ t æ t _ f aɪ v":"<crying>",
                "æ t æ t _ s ɪ k s":"<eating>",
                "æ t æ t _ s ɛ v ə n":"<filler>",
                "æ t æ t _ eɪ t":"<greetings>",
                "æ t æ t _ n aɪ n":"<laughter>",
                "æ t æ t _ t ɛ n":"<screaming>",
                "æ t æ t _ ɪ l ɛ v ə n":"<sneezing>",
                "æ t æ t _ t w ɛ l v":"<throat>",
                "æ t æ t _ θ ɜː t iː n":"<yawning>",
                "æ t æ t _ f oːɹ t iː n":"<yelling>"}
    forbidden_words = set(['#%#', 
                           '##%', 
                           '%%#', 
                           '%#%', 
                           '@@0', 
                           '@@1', 
                           '@@2', 
                           '@@3', 
                           '@@4', 
                           '@@5', 
                           '@@6', 
                           '@@7', 
                           '@@8', 
                           '@@9', 
                           '@@10', 
                           '@@11', 
                           '@@12', 
                           '@@13', 
                           '@@14'])

    ds = EarsDataset(args.input_dir, args.meta)
    loader = DataLoader(
        ds, batch_size=args.mega_batch_size,
        shuffle=False, num_workers=args.n_workers,
        collate_fn=ds.collate, drop_last=False
    )
    
    # 1) 기존 vocab 파일에서 phoneme→index 읽기
    existing_map = {}  # phoneme -> index
    if os.path.isfile(vocab_fn):
        with open(vocab_fn, "r", encoding="utf-8") as vf:
            for line in vf:
                idx_str, phn = line.strip().split(maxsplit=1)
                existing_map[phn] = int(idx_str)
    next_idx = max(existing_map.values(), default=-1) + 1
    
    logging.info(f"phonemizing...")
    phn_vocab = set(existing_map.keys())
    all_lens = []
    skip = 0
    for wav_name, text in tqdm.tqdm(ds.entries, desc="phonemizing"):
        save_fn = os.path.join(phn_save_root, wav_name + ".txt")
        if sum(word in forbidden_words for word in text.split(" ")):
            logging.info(f"skip {wav_name}, forbidden words: {text}")
            skip += 1
            continue
        for k, v in punc2sym.items():
            text = text.replace(k, v)
        phn = tokenize_text(text_tokenizer, text)
        phn_seq = " ".join(phn)
        for k, v in word2sym.items():
            phn_seq = phn_seq.replace(k, v)
        phn_vocab.update(phn_seq.split(" "))
        all_lens.append(len(phn_seq.split(" ")))
        with open(save_fn, "w") as f:
            f.write(phn_seq)
    logging.info(f"Done phonemizing {len(ds.entries)} files, skipped {skip}")

    new_phns = sorted(phn_vocab - set(existing_map.keys()))
    if new_phns:
        logging.info(f"Appending {len(new_phns)} new phonemes to {vocab_fn}")
        with open(vocab_fn, "a", encoding="utf-8") as vf:
            for phn in new_phns:
                vf.write(f"{next_idx} {phn}\n")
                next_idx += 1
    else:
        logging.info("No new phonemes to append.")
    
    for mega_batch in tqdm.tqdm(loader, desc="mega-batches"):
        lengths = np.array(mega_batch['end_time']) - np.array(mega_batch['begin_time'])
        keep = np.where((lengths >= 0.2) & (lengths <= args.len_cap))[0]
        wavs = [mega_batch['audio'][i] for i in keep]
        segs = [mega_batch['segment_id'][i] for i in keep]
        if len(wavs) == 0:
            continue
        if len(segs) == 0:
            continue
        lens = lengths[keep]
        padded = torch.nn.utils.rnn.pad_sequence(wavs, batch_first=True).unsqueeze(1).cuda()
        
        with torch.no_grad():
            codes = model.encode(padded)[0].cpu()
        
        for i, seg in enumerate(segs):
            actual_len = round(lens[i] * args.model_code_sr)
            code_i = codes[i] if not isinstance(codes, list) else codes[i][:, :actual_len]
            # code_i = codes[i][:, :actual_len]
            write_array_to_txt_file(
                code_i.tolist(),
                os.path.join(codes_save_root, seg + ".txt")
            )
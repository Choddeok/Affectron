import os
import torch
import random
import copy
import logging
import shutil
from pathlib import Path
import random
import numpy as np
import json
from glob import glob

class dataset(torch.utils.data.Dataset):
    def __init__(self, args, split):
        super().__init__()
        self.args = args
        self.split = split
        assert self.split in ['train', 'validation', 'test']
        manifest_fn = os.path.join(self.args.manifest_name, self.split+".txt")

        with open(manifest_fn, "r") as rf:
            data = [l.strip().split("\t") for l in rf.readlines()]
        lengths_list = [int(item[-1]) for item in data]
        self.data = []
        self.lengths_list = []
        for d, l in zip(data, lengths_list):
            if l >= self.args.encodec_sr*self.args.audio_min_length:
                if self.args.drop_long and l > self.args.encodec_sr*self.args.audio_max_length:
                    continue
                self.data.append(d)
                self.lengths_list.append(l)
        logging.info(f"number of data points for {self.split} split: {len(self.lengths_list)}")

        # phoneme vocabulary
        vocab_fn = os.path.join(self.args.dataset_dir,"vocab.txt")
        shutil.copy(vocab_fn, os.path.join(self.args.exp_dir, "vocab.txt"))
        with open(vocab_fn, "r") as f:
            temp = [l.strip().split(" ") for l in f.readlines() if len(l) != 0]
            self.phn2num = {item[1]:int(item[0]) for item in temp}
        
        self.symbol_set = set(["<SIL>", "<MUSIC>", "<NOISE>", "<OTHER>"])
    
    def __len__(self):
        return len(self.lengths_list)
    
    def _load_phn_enc(self, index):
        item = self.data[index]
        base_name = Path(item[1]).stem
        pf   = os.path.join(self.args.dataset_dir, self.args.phn_folder_name, item[1] + ".txt")
        ef   = os.path.join(self.args.dataset_dir, self.args.encodec_folder_name, item[1] + ".txt")
        n_cb    = self.args.n_codebooks
        phn_seq = open(pf).read().strip()
        with open(ef, "r", encoding="utf-8") as f:
            encos = [l.strip().split() 
                    for k, l in enumerate(f) 
                    if k < n_cb]

        tokens     = phn_seq.split(" ")
        words = []
        current = []
        for t in tokens:
            if t == "_":
                words.append(current); current = []
            else:
                current.append(t)
        words.append(current)                  

        num_nv = random.randint(0, 2)

        if self.args.data_augmentation and num_nv != 0:
            frames_fn   = Path("/dataset") / "TextGrid_txt" / f"{base_name}.frames.txt"
            eecs_fn = Path("/dataset") / "EECSscroe_top10" / f"{base_name}.txt"
            results_dir = "/dataset/sphere_insertion_txt_results"
            with open(frames_fn, "r", encoding="utf-8") as f:
                frames = [int(line.strip()) for line in f if line.strip()]
            num_intervals = len(frames) 
                
            nv_infos = []
            for nv_idx in range(num_nv):
                if self.args.emotion_driven_nv_matching: 
                    text = eecs_fn.read_text().strip()
                    entries = [e for e in text.split("|") if e]
                    paths, scores = [], []

                    for e in entries:
                        p, s = e.rsplit(",", 1)
                        p, s = p.strip(), s.strip()
                        sc = float(s)
                        paths.append(p)
                        scores.append(sc)

                    arr = np.asarray(scores, dtype=np.float64)
                    mask = np.isfinite(arr)
                    arr = arr[mask]
                    paths = [p for p, m in zip(paths, mask) if m]

                    tau = 0.7
                    tau = max(float(tau), 1e-6)

                    z = arr / tau
                    z = z - np.max(z)          
                    probs = np.exp(z)
                    probs = probs / probs.sum()

                    idx = int(np.random.choice(len(paths), p=probs))
                    selected = paths[idx]
                    nv_name  = Path(selected).stem
                    
                    if self.args.emotion_aware_routing: 
                        txt_fn = os.path.join(results_dir, f"{base_name}_{nv_name}.txt")
                        with open(txt_fn, "r", encoding="utf-8") as f:
                            text = f.read().strip()
                        changes_arr = np.fromstring(text, sep=',', dtype=np.float64) if text else np.array([], dtype=np.float64)
                        changes_arr = np.where(np.isfinite(changes_arr), changes_arr, np.inf)
                        if self.args.min_based_position_select:
                            word_idx = random.randrange(changes_arr)
                        else:
                            size = changes_arr.size
                            k = min(5, size)
                            topk_idx = np.argpartition(changes_arr, k - 1)[:k]
                            topk_vals = changes_arr[topk_idx]

                            tau = 0.7
                            tau = max(tau, 1e-6)
                            z = -topk_vals / tau
                            z = z - np.max(z)
                            exp_z = np.exp(z)
                            probs = exp_z / exp_z.sum()

                            choice_local = int(np.random.choice(len(topk_idx), p=probs))
                            word_idx = int(topk_idx[choice_local])
                            
                        frame_idx = frames[word_idx]
                    else:
                        word_idx = random.randrange(num_intervals)
                        frame_idx = frames[word_idx]
                        
                else:
                    spk_name = base_name.split("_")[0]
                    nvs_dir = "/dataset/NVs"
                    pattern = os.path.join(nvs_dir, f"{spk_name}*.wav")
                    wav_files = glob(pattern)
                    selected_wav = random.choice(wav_files)
                    nv_name = Path(selected_wav).stem
                    
                    word_idx = random.randrange(num_intervals)
                    frame_idx = frames[word_idx]
                    
                pf_nv = os.path.join(self.args.dataset_dir, self.args.phn_folder_name,    nv_name + ".wav.txt")
                ef_nv = os.path.join(self.args.dataset_dir, self.args.encodec_folder_name, nv_name + ".wav.txt")

                phn_nv_seq = open(pf_nv).read().strip().split()
                with open(ef_nv, "r", encoding="utf-8") as f_nv:
                    encos_nv = [l.strip().split() 
                                for k, l in enumerate(f_nv) 
                                if k < n_cb]
                nv_infos.append((nv_name, word_idx, frame_idx, phn_nv_seq, encos_nv))    
            
            if num_nv == 1:
                (nv_name, word_idx, frame_idx, phn_nv_seq, encos_nv) = nv_infos[0]
                new_words = []
                for i,w in enumerate(words):
                    if i == word_idx:
                        new_words.append(phn_nv_seq)
                    new_words.append(w)
                if word_idx == len(words):
                    new_words.append(phn_nv_seq)

                new_tokens = []
                for i,w in enumerate(new_words):
                    new_tokens.extend(w)
                    if i < len(new_words)-1:
                        new_tokens.append("_")
                
                new_encos_str  = []
                for cb in range(n_cb):
                    main_tokens = encos[cb]
                    nv_tokens   = encos_nv[cb]
                    combined = main_tokens[:frame_idx] + nv_tokens + main_tokens[frame_idx:]
                    new_encos_str.append(combined)
                    
                x = [self.phn2num[t] for t in new_tokens 
                    if t not in self.symbol_set]
                if self.args.special_first:
                    y = [
                        [int(n) + self.args.n_special for n in token_list]
                        for token_list in new_encos_str
                    ]
                else:
                    y = [
                        [int(n) for n in token_list]
                        for token_list in new_encos_str
                    ]
                nv_len = len(encos_nv[0])
                center = frame_idx + nv_len // 2 
                y_mask_center = int(center)
                return x, y, y_mask_center
            
            else:
                nv_infos.sort(key=lambda x: x[1])
                
                new_words = words[:]
                new_encos_str = [tokens[:] for tokens in encos]
                
                total_frame_shift = 0
                word_shift = 0
                for (nv_name, word_idx, frame_idx, phn_nv_seq, encos_nv) in nv_infos:
                    adj_word_idx = word_idx + word_shift
                    adj_frame_idx = frame_idx + total_frame_shift
                    new_words.insert(adj_word_idx, phn_nv_seq)

                    for cb in range(n_cb):
                        nv_tokens = encos_nv[cb]
                        main_tokens = new_encos_str[cb]
                        combined = main_tokens[:adj_frame_idx] + nv_tokens + main_tokens[adj_frame_idx:]
                        new_encos_str[cb] = combined
                    word_shift += 1
                    total_frame_shift += len(encos_nv[0])

                new_tokens = []
                for i, w in enumerate(new_words):
                    new_tokens.extend(w)
                    if i < len(new_words) - 1:
                        new_tokens.append("_")

                x = [self.phn2num[t] for t in new_tokens if t not in self.symbol_set]
                if self.args.special_first:
                    y = [[int(n) + self.args.n_special for n in token_list] for token_list in new_encos_str]
                else:
                    y = [[int(n) for n in token_list] for token_list in new_encos_str]

                interval_centers = []
                total_shift = 0
                for (_, _, frame_idx, _, encos_nv) in nv_infos:
                    st = frame_idx + total_shift
                    ed = st + len(encos_nv[0])
                    center = (st + ed) // 2
                    interval_centers.append(int(center))
                    total_shift += len(encos_nv[0])
                    
                y_mask_center  = random.choice(interval_centers)
                return x, y, y_mask_center 

        else:
            # y_mask_center = 0
            try:
                with open(pf, "r") as p, open(ef, "r") as e:
                    phns = [l.strip() for l in p.readlines()]
                    assert len(phns) == 1, phns
                    x = [self.phn2num[item] for item in phns[0].split(" ") if item not in self.symbol_set] # drop ["<SIL>", "<MUSIC>", "<NOISE>", "<OTHER>"], as they are not in training set annotation
                    encos = [l.strip().split() for k, l in enumerate(e.readlines()) if k < self.args.n_codebooks]
                    
                    assert len(encos) == self.args.n_codebooks, ef
                    if self.args.special_first:
                        y = [[int(n)+self.args.n_special for n in l] for l in encos]
                    else:
                        y = [[int(n) for n in l] for l in encos]
            except Exception as e:
                logging.info(f"loading failed for {pf} and {ef}, maybe files don't exist or are corrupted")
                logging.info(f"error message: {e}")
                return [], [[]], 0
            y_len = len(y[0])
            y_mask_center = random.randint(1, y_len-1-self.args.mask_len_min)

            return x, y, y_mask_center

    def __getitem__(self, index):
        x, y, y_mask_center = self._load_phn_enc(index)
        x_len, y_len = len(x), len(y[0])
        
        if x_len == 0 or y_len == 0:
            return {
            "x": None, 
            "x_len": None, 
            "y": None, 
            "y_len": None, 
            "y_mask_center": None
            }
        
        if self.args.drop_long:
            while x_len > self.args.text_max_length or y_len > self.args.encodec_sr*self.args.audio_max_length:
                index = random.choice(range(len(self))) # regenerate an index
                x, y, y_mask_center = self._load_phn_enc(index)
                x_len, y_len = len(x), len(y[0])

        return {
            "x": torch.LongTensor(x), 
            "x_len": x_len, 
            "y": torch.LongTensor(y), 
            "y_len": y_len,
            "y_mask_center": y_mask_center
            }

    def collate(self, batch):
        out = {key:[] for key in batch[0]}
        for item in batch:
            if item['x'] == None: # deal with load failure
                continue
            for key, val in item.items():
                out[key].append(val)
        res = {}
        if self.args.pad_x:
            res["x"] = torch.stack(out["x"], dim=0)
        else:
            res["x"] = torch.nn.utils.rnn.pad_sequence(out["x"], batch_first=True, padding_value=self.args.text_pad_token)
        res["x_lens"] = torch.LongTensor(out["x_len"])
        if self.args.dynamic_batching:
            if out['y'][0].ndim==2:
                res['y'] = torch.nn.utils.rnn.pad_sequence([item.transpose(1,0) for item in out['y']],padding_value=self.args.audio_pad_token)
                res['y'] = res['y'].permute(1,2,0) # T B K -> B K T
            else:
                assert out['y'][0].ndim==1, out['y'][0].shape
                res['y'] = torch.nn.utils.rnn.pad_sequence(out['y'], batch_first=True, padding_value=self.args.audio_pad_token)
        else:
            res['y'] = torch.stack(out['y'], dim=0)
        res["y_lens"] = torch.LongTensor(out["y_len"])
        res["y_mask_center"] = torch.LongTensor(out["y_mask_center"])
        res["text_padding_mask"] = torch.arange(res['x'][0].shape[-1]).unsqueeze(0) >= res['x_lens'].unsqueeze(1)
        res["audio_padding_mask"] = torch.arange(res['y'][0].shape[-1]).unsqueeze(0) >= res['y_lens'].unsqueeze(1)
        return res  
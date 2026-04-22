import os, sys
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from scipy.signal import butter, filtfilt, iirnotch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import config

def process_mnist(raw_file_path, output_dir):
    data_list, labels_list = [], []
    channel_map = {ch: i for i, ch in enumerate(config.EPOC_CHANNELS)}
    event_data = {}
    with open(raw_file_path, 'r') as f:
        for line in tqdm(f):
            parts = line.strip().split('\t')
            if len(parts) < 7: continue
            device = parts[2]
            if device != "EP": continue
            event_id, channel, code = parts[1], parts[3], int(parts[4])
            if code == -1: continue
            raw_vals = [float(x) for x in parts[6].split(',')]
            if len(raw_vals) > config.SEQ_LEN:
                start = (len(raw_vals) - config.SEQ_LEN) // 2
                raw_vals = raw_vals[start : start + config.SEQ_LEN]
            elif len(raw_vals) < config.SEQ_LEN:
                raw_vals = raw_vals + [0.0] * (config.SEQ_LEN - len(raw_vals))
            if event_id not in event_data:
                event_data[event_id] = {'label': code, 'channels': {}}
            event_data[event_id]['channels'][channel] = raw_vals
    for eid, info in tqdm(event_data.items(), desc="Assembling events"):
        if len(info['channels']) == 14:
            ch_array = np.zeros((14, config.SEQ_LEN), dtype=np.float32)
            for ch_name, values in info['channels'].items():
                if ch_name in channel_map:
                    ch_array[channel_map[ch_name]] = values
            data_list.append(ch_array)
            labels_list.append(info['label'])
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "eeg_signals.npy"), np.array(data_list))
    np.save(os.path.join(output_dir, "labels.npy"), np.array(labels_list))

def apply_filters(data, fs=config.EEG_SAMPLING_RATE):
    try:
        nyq = 0.5 * fs
        low = config.EEG_BANDPASS_FREQ[0] / nyq
        high = config.EEG_BANDPASS_FREQ[1] / nyq
        low = max(low, 1e-4)
        high = min(high, 1.0 - 1e-4)
        b, a = butter(4, [low, high], btype='band')
        padlen = 3 * max(len(a), len(b))
        if data.shape[-1] <= padlen:
            return None
        filtered = filtfilt(b, a, data, axis=-1)
        if not np.isfinite(filtered).all():
            return None
        w0 = config.EEG_NOTCH_FREQ / nyq
        w0 = max(0.01, min(w0, 0.99))
        b2, a2 = iirnotch(w0, 30.0)
        filtered = filtfilt(b2, a2, filtered, axis=-1)
        if not np.isfinite(filtered).all():
            return None
        return filtered
    except Exception:
        return None

def _parse_eeg_csv(fpath, channel_map, seq_len, artifact_threshold):
    ch_array = np.zeros((len(channel_map), seq_len), dtype=np.float32)
    found_channels = 0
    with open(fpath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            if ',' in line:
                line_parts = line.split(',')
            else:
                line_parts = line.split('\t')
            if len(line_parts) < 2: continue
            ch_name = line_parts[0].strip()
            if '.' in ch_name:
                ch_name = ch_name.split('.')[-1]
            if ch_name not in channel_map: continue
            vals = [float(x) for x in line_parts[1:] if x.strip()]
            if not vals: continue
            if len(vals) > seq_len:
                start = (len(vals) - seq_len) // 2
                vals = vals[start : start + seq_len]
            elif len(vals) < seq_len:
                vals = vals + [0.0] * (seq_len - len(vals))
            vals_np = np.array(vals, dtype=np.float32)
            max_amp = np.nanmax(np.abs(vals_np))
            if max_amp > artifact_threshold:
                return None, -1
            ch_array[channel_map[ch_name]] = vals_np
            found_channels += 1
    return ch_array, found_channels

def process_imagenet(csv_dir, output_dir, image_dir=None, word_report_path=None):
    csv_files = glob.glob(os.path.join(csv_dir, "**", "*.csv"), recursive=True)
    if not csv_files: return
    data_list, labels_list, imgs_list = [], [], []
    synset_to_id, id_to_word = {}, {}
    if word_report_path and os.path.exists(word_report_path):
        with open(word_report_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    word, synset = parts[0], parts[2]
                    if synset not in synset_to_id:
                        idx = len(synset_to_id)
                        synset_to_id[synset] = idx
                        id_to_word[idx] = word
    channel_map = {ch: i for i, ch in enumerate(config.INSIGHT_CHANNELS)}
    if image_dir and not os.path.exists(image_dir):
        image_dir = None
    for fpath in tqdm(csv_files, desc="CSVs"):
        fname = os.path.basename(fpath)
        parts = fname.split('_')
        if len(parts) < 5: continue
        synset = None
        img_id = None
        for p in parts:
            if p.startswith('n') and len(p) > 5 and p[1:].isdigit():
                synset = p
                idx = parts.index(p)
                if idx + 1 < len(parts):
                    img_id = parts[idx + 1].split('.')[0]
                break
        if not synset: continue
        if synset not in synset_to_id:
            synset_to_id[synset] = len(synset_to_id)
        img_array = None
        if image_dir:
            possible_paths = [
                os.path.join(image_dir, synset, f"{synset}_{img_id}.JPEG"),
                os.path.join(image_dir, synset, f"{synset}_{img_id}.jpg"),
                os.path.join(image_dir, f"{synset}_{img_id}.JPEG"),
                os.path.join(image_dir, f"{synset}_{img_id}.jpg"),
            ]
            if img_id and '.' in img_id:
                possible_paths.append(os.path.join(image_dir, synset, img_id))
                possible_paths.append(os.path.join(image_dir, img_id))
            for p in possible_paths:
                if os.path.exists(p):
                    try:
                        img = Image.open(p).convert('RGB')
                        img = img.resize((config.IMAGE_SIZE, config.IMAGE_SIZE))
                        img_array = np.array(img, dtype=np.uint8)
                    except Exception:
                        pass
                    break
            if img_array is None: continue
        try:
            ch_array, found_channels = _parse_eeg_csv(fpath, channel_map, config.SEQ_LEN, config.EEG_ARTIFACT_THRESHOLD)
        except Exception: continue
        if found_channels == -1 or found_channels < 3: continue
        filtered = apply_filters(ch_array)
        if filtered is None: continue
        ch_array = filtered
        post_max = np.nanmax(np.abs(ch_array))
        if post_max > 150.0: continue
        means = ch_array.mean(axis=1, keepdims=True)
        stds = ch_array.std(axis=1, keepdims=True) + 1e-6
        ch_array = (ch_array - means) / stds
        data_list.append(ch_array.astype(np.float32))
        labels_list.append(synset_to_id[synset])
        if image_dir and img_array is not None:
            imgs_list.append(img_array)
    if not data_list: return
    os.makedirs(output_dir, exist_ok=True)
    unique_old = sorted(set(labels_list))
    remap = {old: new for new, old in enumerate(unique_old)}
    labels_list = [remap[l] for l in labels_list]
    synset_to_id = {k: remap[v] for k, v in synset_to_id.items() if v in remap}
    np.save(os.path.join(output_dir, "eeg_signals.npy"), np.array(data_list))
    np.save(os.path.join(output_dir, "labels.npy"), np.array(labels_list, dtype=np.int64))
    if imgs_list:
        np.save(os.path.join(output_dir, "images.npy"), np.array(imgs_list))
    import json
    meta = {
        "synset_to_id": synset_to_id,
        "id_to_word": id_to_word,
    }
    with open(os.path.join(output_dir, "metadata.json"), 'w') as jf:
        json.dump(meta, jf)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["mnist", "imagenet"], required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--image_dir", type=str, default=None)
    parser.add_argument("--word_report", type=str, default=None)
    args = parser.parse_args()
    if args.mode == "mnist":
        process_mnist(args.input, args.output)
    else:
        process_imagenet(args.input, args.output, args.image_dir, args.word_report)
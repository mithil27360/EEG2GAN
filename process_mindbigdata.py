import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import config

def process_mnist(raw_file_path, output_dir):
    print(f"Processing MNIST EEG from: {raw_file_path}")
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
    print(f"Saved {len(data_list)} MNIST samples to {output_dir}")

def process_imagenet(csv_dir, output_dir, image_dir=None, word_report_path=None):
    print(f"Processing ImageNet EEG from: {csv_dir}")
    csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
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
    for fpath in tqdm(csv_files):
        fname = os.path.basename(fpath)
        parts = fname.split('_')
        synset, img_id = parts[3], parts[4]
        if synset not in synset_to_id:
            synset_to_id[synset] = len(synset_to_id)
        img_array = None
        if image_dir:
            img_path = os.path.join(image_dir, f"{synset}_{img_id}.JPEG")
            if not os.path.exists(img_path):
                alt_paths = glob.glob(os.path.join(image_dir, "**", f"{synset}_{img_id}.JPEG"), recursive=True)
                if alt_paths: img_path = alt_paths[0]
            if os.path.exists(img_path):
                try:
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize((config.IMAGE_SIZE, config.IMAGE_SIZE))
                    img_array = np.array(img, dtype=np.uint8)
                except: continue
        if image_dir and img_array is None: continue
        try:
            with open(fpath, 'r') as f:
                ch_array = np.zeros((5, config.SEQ_LEN), dtype=np.float32)
                found_channels = 0
                for line in f:
                    line_parts = line.strip().split(',')
                    ch_name = line_parts[0]
                    if ch_name in channel_map:
                        vals = [float(x) for x in line_parts[1:]]
                        if len(vals) > config.SEQ_LEN:
                            start = (len(vals) - config.SEQ_LEN) // 2
                            vals = vals[start : start + config.SEQ_LEN]
                        elif len(vals) < config.SEQ_LEN:
                            vals = vals + [0.0] * (config.SEQ_LEN - len(vals))
                        ch_array[channel_map[ch_name]] = vals
                        found_channels += 1
                if found_channels == 5:
                    data_list.append(ch_array)
                    labels_list.append(synset_to_id[synset])
                    if image_dir: imgs_list.append(img_array)
        except: continue
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "eeg_signals.npy"), np.array(data_list))
    np.save(os.path.join(output_dir, "labels.npy"), np.array(labels_list))
    if image_dir: np.save(os.path.join(output_dir, "images.npy"), np.array(imgs_list))
    meta = {"synset_to_id": synset_to_id, "id_to_word": id_to_word}
    with open(os.path.join(output_dir, "metadata.json"), 'w') as jf:
        import json
        json.dump(meta, jf)
    print(f"Saved {len(data_list)} samples to {output_dir}")

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

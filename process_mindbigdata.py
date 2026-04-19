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
    csv_files = glob.glob(os.path.join(csv_dir, "**", "*.csv"), recursive=True)
    if not csv_files:
        print(f"Error: No CSV files found in {csv_dir}")
        return

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
    missing_images_count = 0
    found_images_count = 0
    skipped_csv_count = 0

    for fpath in tqdm(csv_files, desc="Processing CSVs"):
        fname = os.path.basename(fpath)
        parts = fname.split('_')
        if len(parts) < 5: 
            skipped_csv_count += 1
            continue
        
        # Filename: MindBigData_Imagenet_Insight_SYNSEID_IMAGEID_CHANNEL_EVENT.csv
        # OR: MindBigData_Imagenet_Insight_SYNSEID_IMAGEID.csv
        # Let's try to find synset and image ID by looking for nXXXX
        synset = None
        img_id = None
        for p in parts:
            if p.startswith('n') and len(p) > 5 and p[1:].isdigit():
                synset = p
                # The next part is usually the image ID
                idx = parts.index(p)
                if idx + 1 < len(parts):
                    img_id = parts[idx+1].split('.')[0]
                break
        
        if not synset:
            skipped_csv_count += 1
            continue
        if synset not in synset_to_id:
            synset_to_id[synset] = len(synset_to_id)
            
        img_array = None
        if image_dir:
            if not os.path.exists(image_dir):
                print(f"Error: image_dir {image_dir} does not exist!")
                image_dir = None
            else:
                # Try multiple possible naming conventions for ImageNet
                possible_paths = [
                    os.path.join(image_dir, synset, f"{synset}_{img_id}.JPEG"),
                    os.path.join(image_dir, synset, f"{synset}_{img_id}.jpg"),
                    os.path.join(image_dir, f"{synset}_{img_id}.JPEG"),
                    os.path.join(image_dir, f"{synset}_{img_id}.jpg"),
                ]
                # Also try just the image ID if it looks like a full filename
                if '.' in img_id:
                    possible_paths.append(os.path.join(image_dir, synset, img_id))
                    possible_paths.append(os.path.join(image_dir, img_id))

                for p in possible_paths:
                    if os.path.exists(p):
                        try:
                            img = Image.open(p).convert('RGB')
                            img = img.resize((config.IMAGE_SIZE, config.IMAGE_SIZE))
                            img_array = np.array(img, dtype=np.uint8)
                            found_images_count += 1
                            break
                        except: continue
        
        if image_dir and img_array is None:
            missing_images_count += 1
            continue

        try:
            with open(fpath, 'r') as f:
                ch_array = np.zeros((5, config.SEQ_LEN), dtype=np.float32)
                found_channels = 0
                for line in f:
                    line = line.strip()
                    if not line: continue
                    # Robust delimiter handling
                    if ',' in line:
                        line_parts = line.split(',')
                    else:
                        line_parts = line.split('\t')
                        
                    if len(line_parts) < 2: continue
                    ch_name = line_parts[0].strip()
                    # Remove prefixes like 'EEG.' if present
                    if '.' in ch_name: ch_name = ch_name.split('.')[-1]
                    
                    if ch_name in channel_map:
                        vals = [float(x) for x in line_parts[1:] if x.strip()]
                        if len(vals) > config.SEQ_LEN:
                            start = (len(vals) - config.SEQ_LEN) // 2
                            vals = vals[start : start + config.SEQ_LEN]
                        elif len(vals) < config.SEQ_LEN:
                            vals = vals + [0.0] * (config.SEQ_LEN - len(vals))
                        ch_array[channel_map[ch_name]] = vals
                        found_channels += 1
                
                if found_channels >= 3: # Allow partial if most channels found
                    data_list.append(ch_array)
                    labels_list.append(synset_to_id[synset])
                    if image_dir: imgs_list.append(img_array)
                else:
                    skipped_csv_count += 1
        except: 
            skipped_csv_count += 1
            continue

    if missing_images_count > 0:
        print(f"Note: {missing_images_count} samples skipped because images were not found in {image_dir}")
    if skipped_csv_count > 0:
        print(f"Note: {skipped_csv_count} CSV files skipped due to parsing errors or missing channels.")
    
    if not data_list:
        print("Error: No samples were successfully processed!")
        if image_dir:
            print("Tip: Try running without --image_dir to see if EEG data alone can be processed.")
        return

    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "eeg_signals.npy"), np.array(data_list))
    np.save(os.path.join(output_dir, "labels.npy"), np.array(labels_list))
    if len(imgs_list) > 0: 
        np.save(os.path.join(output_dir, "images.npy"), np.array(imgs_list))
    
    meta = {"synset_to_id": synset_to_id, "id_to_word": id_to_word}
    with open(os.path.join(output_dir, "metadata.json"), 'w') as jf:
        import json
        json.dump(meta, jf)
    print(f"Successfully saved {len(data_list)} samples to {output_dir}")

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

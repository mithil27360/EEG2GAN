import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from scipy.signal import butter, filtfilt, iirnotch
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

def apply_filters(data, fs=config.EEG_SAMPLING_RATE):
    """Apply bandpass + notch filter. Returns None if filtering fails or produces NaN."""
    try:
        nyq = 0.5 * fs
        low  = config.EEG_BANDPASS_FREQ[0] / nyq
        high = config.EEG_BANDPASS_FREQ[1] / nyq
        # Clamp to valid range
        low  = max(low,  1e-4)
        high = min(high, 1.0 - 1e-4)
        b, a = butter(4, [low, high], btype='band')

        # filtfilt needs >= padlen samples; padlen = 3*max(len(a),len(b))
        padlen = 3 * max(len(a), len(b))
        if data.shape[-1] <= padlen:
            return None

        filtered = filtfilt(b, a, data, axis=-1)
        if not np.isfinite(filtered).all():
            return None

        # Notch filter
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
    """
    Parse a single EEG CSV file.
    Returns (ch_array, found_channels) or raises ValueError with reason.
    found_channels == -1 means artifact rejection.
    """
    ch_array = np.zeros((len(channel_map), seq_len), dtype=np.float32)
    found_channels = 0

    with open(fpath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Robust delimiter handling
            if ',' in line:
                line_parts = line.split(',')
            else:
                line_parts = line.split('\t')

            if len(line_parts) < 2:
                continue
            ch_name = line_parts[0].strip()
            # Remove prefixes like 'EEG.' if present
            if '.' in ch_name:
                ch_name = ch_name.split('.')[-1]

            if ch_name not in channel_map:
                continue

            vals = [float(x) for x in line_parts[1:] if x.strip()]
            if not vals:
                continue

            if len(vals) > seq_len:
                start = (len(vals) - seq_len) // 2
                vals = vals[start : start + seq_len]
            elif len(vals) < seq_len:
                vals = vals + [0.0] * (seq_len - len(vals))

            vals_np = np.array(vals, dtype=np.float32)

            # Pre-filter artifact rejection (raw amplitude)
            max_amp = np.nanmax(np.abs(vals_np))
            if max_amp > artifact_threshold:
                return None, -1  # Artifact rejected

            ch_array[channel_map[ch_name]] = vals_np
            found_channels += 1

    return ch_array, found_channels


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
    skipped_csv_count    = 0
    artifact_count       = 0
    filter_fail_count    = 0
    low_channel_count    = 0
    parse_error_count    = 0

    # Validate image_dir upfront
    if image_dir and not os.path.exists(image_dir):
        print(f"Warning: image_dir '{image_dir}' does not exist — ignoring image loading.")
        image_dir = None

    for fpath in tqdm(csv_files, desc="Processing CSVs"):
        fname = os.path.basename(fpath)
        parts = fname.split('_')
        if len(parts) < 5:
            skipped_csv_count += 1
            continue

        # Filename: MindBigData_Imagenet_Insight_SYNSEID_IMAGEID_CHANNEL_EVENT.csv
        synset  = None
        img_id  = None
        for p in parts:
            if p.startswith('n') and len(p) > 5 and p[1:].isdigit():
                synset = p
                idx = parts.index(p)
                if idx + 1 < len(parts):
                    img_id = parts[idx + 1].split('.')[0]
                break

        if not synset:
            skipped_csv_count += 1
            continue
        if synset not in synset_to_id:
            synset_to_id[synset] = len(synset_to_id)

        # --- Image loading ---
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
                    break  # stop after first path match regardless of open success

            if img_array is None:
                missing_images_count += 1
                continue  # skip sample if image required but not found

        # --- EEG parsing ---
        try:
            ch_array, found_channels = _parse_eeg_csv(
                fpath, channel_map, config.SEQ_LEN, config.EEG_ARTIFACT_THRESHOLD
            )
        except Exception as e:
            parse_error_count += 1
            continue

        if found_channels == -1:
            artifact_count += 1
            continue

        if found_channels < 3:
            low_channel_count += 1
            continue

        # --- Signal filtering ---
        filtered = apply_filters(ch_array)
        if filtered is None:
            filter_fail_count += 1
            continue
        ch_array = filtered

        # Channel-wise Z-score normalization
        means = ch_array.mean(axis=1, keepdims=True)
        stds  = ch_array.std(axis=1, keepdims=True) + 1e-6
        ch_array = (ch_array - means) / stds

        data_list.append(ch_array.astype(np.float32))
        labels_list.append(synset_to_id[synset])
        if image_dir and img_array is not None:
            imgs_list.append(img_array)

    # Diagnostic summary
    print(f"\n--- Processing Summary ---")
    print(f"  Total CSVs found      : {len(csv_files)}")
    print(f"  Missing images        : {missing_images_count}")
    print(f"  Bad filename format   : {skipped_csv_count}")
    print(f"  Artifact rejected     : {artifact_count}")
    print(f"  Filter failed (NaN)   : {filter_fail_count}")
    print(f"  Too few channels (<3) : {low_channel_count}")
    print(f"  Parse errors          : {parse_error_count}")
    print(f"  Successfully saved    : {len(data_list)}")
    print(f"--------------------------\n")

    if not data_list:
        print("Error: No samples were successfully processed!")
        if image_dir:
            print("Tip: The image directory was specified but 0 samples passed all filters.")
            print("     Try running without --image_dir first to verify EEG parsing works.")
        return

    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "eeg_signals.npy"), np.array(data_list))
    np.save(os.path.join(output_dir, "labels.npy"),      np.array(labels_list))
    if imgs_list:
        np.save(os.path.join(output_dir, "images.npy"), np.array(imgs_list))

    meta = {"synset_to_id": synset_to_id, "id_to_word": id_to_word}
    with open(os.path.join(output_dir, "metadata.json"), 'w') as jf:
        import json
        json.dump(meta, jf)
    print(f"Successfully saved {len(data_list)} samples to {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",       choices=["mnist", "imagenet"], required=True)
    parser.add_argument("--input",      type=str, required=True)
    parser.add_argument("--output",     type=str, required=True)
    parser.add_argument("--image_dir",  type=str, default=None)
    parser.add_argument("--word_report",type=str, default=None)
    args = parser.parse_args()
    if args.mode == "mnist":
        process_mnist(args.input, args.output)
    else:
        process_imagenet(args.input, args.output, args.image_dir, args.word_report)

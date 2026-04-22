#!/bin/bash

echo "--- Pre-Flight Check ---"
BASE_DIR="/kaggle/working/EEG2GAN"
DATA_DIR="/kaggle/working/data/mindbigdata_imagenet"
ZIP_FILE="/kaggle/working/EEG2GAN/eeg_data.zip"

if [ ! -d "/kaggle/working/raw_eeg" ]; then
    echo " Unzipping EEG data..."
    mkdir -p /kaggle/working/raw_eeg
    if [ -f "$ZIP_FILE" ]; then
        unzip -q "$ZIP_FILE" -d /kaggle/working/raw_eeg
    elif [ -f "eeg_data.zip" ]; then
        unzip -q "eeg_data.zip" -d /kaggle/working/raw_eeg
    else
        echo " Error: eeg_data.zip not found!"
        exit 1
    fi
fi

echo " Locating CSV files..."
CSV_ROOT=$(find /kaggle/working/raw_eeg -name "*.csv" | head -n 1 | xargs -r dirname)

if [ -z "$CSV_ROOT" ]; then
    echo " Error: No CSV files found in /kaggle/working/raw_eeg"
    exit 1
fi
echo " Found CSVs in: $CSV_ROOT"

echo " Detecting ImageNet..."
IMAGENET_PATH=$(python -c "import config; print(config.IMAGENET_DIR)")

if [ ! -d "$IMAGENET_PATH" ]; then
    echo " Warning: ImageNet path from config ($IMAGENET_PATH) not found."
    echo " Checking standard Kaggle paths..."
    if [ -d "/kaggle/input/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train" ]; then
        IMAGENET_PATH="/kaggle/input/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train"
    elif [ -d "/kaggle/input/imagenet-1k/train" ]; then
        IMAGENET_PATH="/kaggle/input/imagenet-1k/train"
    else
        echo " Warning: ImageNet not found. Will attempt EEG-only processing."
        IMAGENET_PATH=""
    fi
fi
echo " Using ImageNet path: $IMAGENET_PATH"

echo " Processing EEG Data..."
python process_mindbigdata.py --mode imagenet --input "$CSV_ROOT" --output "$DATA_DIR" --image_dir "$IMAGENET_PATH"

echo "--- Final Check ---"
if [ -f "$DATA_DIR/eeg_signals.npy" ]; then
    echo " Success: processed data saved to $DATA_DIR"
    ls -lh "$DATA_DIR"
else
    echo " Error: Data processing failed. No .npy files found in $DATA_DIR"
fi
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
CSV_ROOT=$(find /kaggle/working/raw_eeg -name "*.csv" -print -quit | xargs dirname)

if [ -z "$CSV_ROOT" ]; then
    echo " Error: No CSV files found in /kaggle/working/raw_eeg"
    exit 1
fi
echo " Found CSVs in: $CSV_ROOT"

echo " Detecting ImageNet..."
IMAGENET_PATH="/kaggle/input/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train"
if [ ! -d "$IMAGENET_PATH" ]; then
    if [ -d "/kaggle/input/imagenet-1k/train" ]; then
        IMAGENET_PATH="/kaggle/input/imagenet-1k/train"
    elif [ -d "/kaggle/input/imagenet/train" ]; then
        IMAGENET_PATH="/kaggle/input/imagenet/train"
    else
        echo " Warning: ImageNet training set not found at standard Kaggle paths."
        echo "   Please ensure the ImageNet dataset is added to your notebook."
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

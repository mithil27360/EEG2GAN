import os

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))

IS_KAGGLE = os.path.exists("/kaggle/input")

if IS_KAGGLE:
    if os.path.exists("/kaggle/input/eeg2image-dataset"):
        DATA_DIR = "/kaggle/input/eeg2image-dataset"
    else:
        DATA_DIR = "/kaggle/working/data"
    CHECKPOINT_DIR = "/kaggle/working/checkpoints"
    OUTPUT_DIR     = "/kaggle/working/outputs"
    FIGURES_DIR    = "/kaggle/working/figures"
else:
    DATA_DIR       = os.path.join(BASE_DIR, "data")
    CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
    OUTPUT_DIR     = os.path.join(BASE_DIR, "outputs")
    FIGURES_DIR    = os.path.join(BASE_DIR, "figures")

# ImageNet train directory — put the verified competition path first
IMAGENET_DIR = "/kaggle/input/competitions/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train"
if not os.path.exists(IMAGENET_DIR):
    for _p in [
        "/kaggle/input/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train",
        "/kaggle/input/imagenet-1k/train",
        "/kaggle/input/imagenet/train",
        os.path.join(DATA_DIR, "imagenet", "train"),
    ]:
        if os.path.exists(_p):
            IMAGENET_DIR = _p
            break

# ── Dataset paths ──────────────────────────────────────────────────────────────
THOUGHTVIZ_EEG_OBJECTS    = os.path.join(DATA_DIR, "thoughtviz", "object", "eeg_signals.npy")
THOUGHTVIZ_LABELS_OBJECTS = os.path.join(DATA_DIR, "thoughtviz", "object", "labels.npy")
THOUGHTVIZ_IMAGES_OBJECTS = os.path.join(DATA_DIR, "thoughtviz", "object", "images.npy")

THOUGHTVIZ_EEG_CHARS      = os.path.join(DATA_DIR, "thoughtviz", "character", "eeg_signals.npy")
THOUGHTVIZ_LABELS_CHARS   = os.path.join(DATA_DIR, "thoughtviz", "character", "labels.npy")
THOUGHTVIZ_IMAGES_CHARS   = os.path.join(DATA_DIR, "thoughtviz", "character", "images.npy")

MINDBIGDATA_EEG    = os.path.join(DATA_DIR, "mindbigdata", "eeg_signals.npy")
MINDBIGDATA_LABELS = os.path.join(DATA_DIR, "mindbigdata", "labels.npy")
MINDBIGDATA_IMAGES = os.path.join(DATA_DIR, "mindbigdata", "images.npy")

MINDBIGDATA_IMAGENET_EEG    = os.path.join(DATA_DIR, "mindbigdata_imagenet", "eeg_signals.npy")
MINDBIGDATA_IMAGENET_LABELS = os.path.join(DATA_DIR, "mindbigdata_imagenet", "labels.npy")
MINDBIGDATA_IMAGENET_IMAGES = os.path.join(DATA_DIR, "mindbigdata_imagenet", "images.npy")
MINDBIGDATA_IMAGENET_META   = os.path.join(DATA_DIR, "mindbigdata_imagenet", "metadata.json")

# ── EEG hardware channels ──────────────────────────────────────────────────────
EPOC_CHANNELS    = ["AF3","F7","F3","FC5","T7","P7","O1","O2","P8","T8","FC6","F4","F8","AF4"]
INSIGHT_CHANNELS = ["AF3","AF4","T7","T8","Pz"]

N_CHANNELS         = 14          # EPOC (Thoughtviz / MindBigData MNIST)
IMAGENET_CHANNELS  = 5           # Insight (MindBigData ImageNet)
SEQ_LEN            = 128

N_CLASSES_OBJECTS = 10
N_CLASSES_CHARS   = 10

# ── Encoder architecture ───────────────────────────────────────────────────────
EMBED_DIM   = 128
N_HEADS     = 4
N_LAYERS    = 3
FF_DIM      = 256
DROPOUT     = 0.1
OUT_DIM     = 256

# ── EEG signal pipeline ────────────────────────────────────────────────────────
EEG_SAMPLING_RATE      = 128        # Emotiv Insight sample rate
EEG_BANDPASS_FREQ      = [1.0, 50.0]
EEG_NOTCH_FREQ         = 50.0
EEG_ARTIFACT_THRESHOLD = 5000.0    # µV ceiling for RAW pre-filter data (sensor disconnect)
EEG_AUG_NOISE_STD      = 0.05
EEG_AUG_SHIFT_MAX      = 8
EEG_AUG_MASK_LEN       = 16
EEG_WINDOW_SIZE        = 128
EEG_WINDOW_STRIDE      = 128       # no overlap — avoids label-mapping edge case
# IMPORTANT: process_mindbigdata.py already Z-scores each channel before saving.
# Setting this to True would double-normalize and collapse inter-sample variance.
EEG_NORMALIZE          = False
BALANCED_SAMPLING      = False     # random shuffle covers many classes better
SAMPLES_PER_CLASS      = 4

# ── Encoder training ───────────────────────────────────────────────────────────
MARGIN           = 0.3
ENC_LR           = 3e-4
ENC_WEIGHT_DECAY = 1e-4
ENC_BATCH_SIZE   = 64             # larger batch → more stable CE gradient with many classes
ENC_EPOCHS       = 500
ENC_PATIENCE     = 150            # 15 non-improving 10-epoch blocks before early stop

# ── GAN architecture ───────────────────────────────────────────────────────────
NOISE_DIM    = 100
EEG_FEAT_DIM = 256
Z_DIM        = NOISE_DIM + EEG_FEAT_DIM

IMAGE_SIZE   = 128
NC           = 3
NGF          = 64
NDF          = 64

# ── GAN training ───────────────────────────────────────────────────────────────
GAN_LR_G       = 0.0001
GAN_LR_D       = 0.0001
BETA1          = 0.0
BETA2          = 0.999
GAN_BATCH_SIZE = 32
GAN_EPOCHS     = 300
LAMBDA_MS      = 2.0
LEAKY_SLOPE    = 0.1

DIFFAUG_POLICY = "color,translation,cutout"

# ── Evaluation ─────────────────────────────────────────────────────────────────
KMEANS_N_INIT = 20
IS_SPLITS     = 10
IS_N_SAMPLES  = 2048
CLIP_MODEL    = "openai/clip-vit-base-patch32"

SEED = 999

# ── Ablation study variants ────────────────────────────────────────────────────
ABLATION_CONFIGS = [
    {"n_layers": 1, "pooling": "mean", "tag": "L1_mean"},
    {"n_layers": 2, "pooling": "mean", "tag": "L2_mean"},
    {"n_layers": 4, "pooling": "mean", "tag": "L4_mean"},
    {"n_layers": 2, "pooling": "cls",  "tag": "L2_cls"},
    {"n_layers": 2, "pooling": "mean", "tag": "L2_noDiffAug", "diffaug": False},
]

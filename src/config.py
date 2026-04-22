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
    DATA_DIR       = os.path.join(BASE_DIR, "results", "data")
    CHECKPOINT_DIR = os.path.join(BASE_DIR, "results", "checkpoints")
    OUTPUT_DIR     = os.path.join(BASE_DIR, "results", "outputs")
    FIGURES_DIR    = os.path.join(BASE_DIR, "results", "figures")

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

EPOC_CHANNELS    = ["AF3","F7","F3","FC5","T7","P7","O1","O2","P8","T8","FC6","F4","F8","AF4"]
INSIGHT_CHANNELS = ["AF3","AF4","T7","T8","Pz"]

N_CHANNELS         = 14
IMAGENET_CHANNELS  = 5
SEQ_LEN            = 128

N_CLASSES_OBJECTS = 10
N_CLASSES_CHARS   = 10

EMBED_DIM   = 64
N_HEADS     = 4
N_LAYERS    = 2
FF_DIM      = 128
DROPOUT     = 0.1
OUT_DIM     = 128

EEG_SAMPLING_RATE      = 128
EEG_BANDPASS_FREQ      = [1.0, 50.0]
EEG_NOTCH_FREQ         = 50.0
EEG_ARTIFACT_THRESHOLD = 5000.0
EEG_AUG_NOISE_STD      = 0.05
EEG_AUG_SHIFT_MAX      = 8
EEG_AUG_MASK_LEN       = 16
EEG_WINDOW_SIZE        = 128
EEG_WINDOW_STRIDE      = 128
EEG_NORMALIZE          = False
BALANCED_SAMPLING      = False
SAMPLES_PER_CLASS      = 4

MARGIN           = 0.3
ENC_LR           = 3e-4
ENC_WEIGHT_DECAY = 1e-4
ENC_BATCH_SIZE   = 32
ENC_EPOCHS       = 500
ENC_PATIENCE     = 50

NOISE_DIM    = 100
EEG_FEAT_DIM = 128
Z_DIM        = NOISE_DIM + EEG_FEAT_DIM

IMAGE_SIZE   = 128
NC           = 3
NGF          = 64
NDF          = 64

GAN_LR_G       = 0.0002
GAN_LR_D       = 0.0002
BETA1          = 0.5
BETA2          = 0.999
GAN_BATCH_SIZE = 16
GAN_EPOCHS     = 200
LAMBDA_MS      = 1.0
LEAKY_SLOPE    = 0.1

DIFFAUG_POLICY = "color,translation,cutout"

KMEANS_N_INIT = 20
IS_SPLITS     = 10
IS_N_SAMPLES  = 2048
CLIP_MODEL    = "openai/clip-vit-base-patch32"

SEED = 999

ABLATION_CONFIGS = [
    {"n_layers": 1, "pooling": "mean", "tag": "L1_mean"},
    {"n_layers": 2, "pooling": "mean", "tag": "L2_mean"},
    {"n_layers": 4, "pooling": "mean", "tag": "L4_mean"},
    {"n_layers": 2, "pooling": "cls",  "tag": "L2_cls"},
    {"n_layers": 2, "pooling": "mean", "tag": "L2_noDiffAug", "diffaug": False},
]
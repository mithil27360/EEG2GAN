import os, sys, json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import Counter
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
import nltk

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

BG = "#ffffff"
BLUE = "#1a6fbf"
GRAY = "#444444"
FONT_SIZE = 9
PALETTE = "Blues"
_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(config.DATA_DIR, "mindbigdata_imagenet")
DEFAULT_CKPT = config.CHECKPOINT_DIR

nltk.download('wordnet', quiet=True)
from nltk.corpus import wordnet as wn

meta = json.load(open(os.path.join(DEFAULT_DATA, "metadata.json")))
id_to_synset = {v: k for k, v in meta["synset_to_id"].items()}

def get_name(lid):
    syn = id_to_synset.get(int(lid), "")
    try:
        ss = wn.synset_from_pos_and_offset('n', int(syn[1:]))
        return ss.lemma_names()[0].replace('_', ' ').title()
    except:
        return f"ID {lid}"

def fig1_stats():
    lbls = np.load(os.path.join(DEFAULT_DATA, "labels.npy"))
    counts = Counter(lbls)
    top_25 = counts.most_common(25)
    names = [get_name(lid) for lid, _ in top_25]
    vals = [count for _, count in top_25]
    plt.figure(figsize=(8, 8), facecolor=BG)
    plt.barh(names[::-1], vals[::-1], color=BLUE, alpha=0.8)
    plt.gca().set_facecolor("#f9f9f9")
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.xlabel("Number of EEG Samples")
    plt.ylabel("ImageNet Class (Synset)")
    plt.tight_layout()
    plt.savefig("fig_dataset_stats.png", dpi=150, bbox_inches='tight')
    plt.close()

def fig2_confusion():
    embs = np.load(os.path.join(DEFAULT_CKPT, "embeddings_transformer_imagenet_main.npy"))
    lbls_full = np.load(os.path.join(DEFAULT_DATA, "labels.npy"))
    lbls = lbls_full[:len(embs)]
    counts = Counter(lbls)
    top_k_classes = [lid for lid, count in counts.most_common(12)]
    mask = np.isin(lbls, top_k_classes)
    X = embs[mask]
    y_true = lbls[mask]
    unique_y = sorted(list(set(y_true)))
    id_map = {old: new for new, old in enumerate(unique_y)}
    y_true_mapped = np.array([id_map[val] for val in y_true])
    n_classes = len(unique_y)
    kmeans = KMeans(n_clusters=n_classes, n_init=10, random_state=42)
    y_pred = kmeans.fit_predict(X)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true_mapped, y_pred, labels=range(n_classes))
    row_ind, col_ind = linear_sum_assignment(-cm)
    new_cm = cm[:, col_ind]
    fig, ax = plt.subplots(figsize=(10, 8), facecolor=BG)
    im = ax.imshow(new_cm, cmap=PALETTE)
    cbar = plt.colorbar(im)
    thresh = new_cm.max() / 2.
    for i in range(new_cm.shape[0]):
        for j in range(new_cm.shape[1]):
            ax.text(j, i, format(new_cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if new_cm[i, j] > thresh else "black")
    ax.set_xticks(range(n_classes))
    ax.set_xticklabels([get_name(l) for l in unique_y], rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(n_classes))
    ax.set_yticklabels([get_name(l) for l in unique_y], fontsize=8)
    plt.xlabel("Assigned Cluster (K-Means + Hungarian)")
    plt.ylabel("Ground Truth Class")
    plt.tight_layout()
    plt.savefig("fig_confusion_matrix.png", dpi=150, bbox_inches='tight')
    plt.close()

def fig3_table():
    data = [
        ["Method", "Architecture", "IS (↑)", "EISC (↑)", "K-Means (↑)", "FID (↓)"],
        ["ThoughtViz (2017)", "CNN-GAN", "4.12", "0.211", "0.082", "312.4"],
        ["LSTM Baseline", "LSTM + DCGAN", "6.15", "0.419", "0.205", "141.4"],
        ["EEG2GAN (Ours)", "Transformer + DCGAN", "7.10", "0.478", "0.206", "128.9"],
    ]
    fig, ax = plt.subplots(figsize=(10, 4), facecolor=BG)
    ax.axis('off')
    plt.title("Quantitative Comparison: Image Quality and Semantic Alignment", 
              fontsize=12, fontweight='bold', pad=20)
    col_widths = [0.22, 0.22, 0.1, 0.1, 0.1, 0.1]
    table = ax.table(cellText=data, loc='center', cellLoc='center', 
                     edges='horizontal', colWidths=col_widths)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.8)
    for i in range(len(data)):
        for j in range(len(data[0])):
            cell = table[(i, j)]
            if i == 0:
                cell.set_text_props(weight='bold', color='white')
                cell.set_facecolor(BLUE)
            elif i == len(data) - 1:
                cell.set_text_props(weight='bold')
                cell.set_facecolor('#e3f2fd')
            if j == 2 or j == 3 or j == 4 or j == 5:
                if i > 0:
                    cell.set_text_props(fontstyle='italic')
    plt.savefig("fig_results_table.png", dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    fig1_stats()
    fig2_confusion()
    fig3_table()

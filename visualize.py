import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TABLEAU_COLORS
from sklearn.manifold import TSNE
import config

os.makedirs(config.FIGURES_DIR, exist_ok=True)
COLORS = list(TABLEAU_COLORS.values())[:10]

def plot_tsne(emb_transformer, emb_lstm, labels, class_names=None, save=True):
    n_classes = len(np.unique(labels))
    if class_names is None:
        class_names = [f"Class {i}" for i in range(n_classes)]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, emb, title in zip(axes, [emb_lstm, emb_transformer], ["LSTM Encoder (Baseline)", "Transformer Encoder (Ours)"]):
        tsne  = TSNE(n_components=2, perplexity=min(30, len(emb)//4), random_state=config.SEED, max_iter=1000)
        proj  = tsne.fit_transform(emb)
        for cls_idx in range(n_classes):
            mask = labels == cls_idx
            ax.scatter(proj[mask, 0], proj[mask, 1], c=COLORS[cls_idx % len(COLORS)], label=class_names[cls_idx], alpha=0.7, s=40)
        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])
    axes[1].legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    path = os.path.join(config.FIGURES_DIR, "fig2_tsne.png")
    if save:
        plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path

def plot_training_curves(g_losses_list, d_losses_list, labels_list, dataset="objects", save=True):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for g_losses, d_losses, label in zip(g_losses_list, d_losses_list, labels_list):
        epochs = range(1, len(g_losses) + 1)
        axes[0].plot(epochs, g_losses, label=label)
        axes[1].plot(epochs, d_losses, label=label)
    axes[0].set_title("Generator Loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].set_title("Discriminator Loss")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(config.FIGURES_DIR, f"fig5_training_curves_{dataset}.png")
    if save:
        plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path

def plot_ablation_bars(config_names, IS_scores, EISC_scores, kmeans_scores, save=True):
    x = np.arange(len(config_names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, IS_scores, width, label="IS ")
    ax.bar(x, EISC_scores, width, label="EISC ")
    ax.bar(x + width, kmeans_scores, width, label="k-means acc ")
    ax.set_xticks(x)
    ax.set_xticklabels(config_names, rotation=20, ha="right")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(config.FIGURES_DIR, "fig4_ablation_bars.png")
    if save:
        plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path

def plot_eisc_vs_is(IS_list, EISC_list, labels_list, save=True):
    fig, ax = plt.subplots(figsize=(7, 5))
    for is_val, eisc_val, label in zip(IS_list, EISC_list, labels_list):
        ax.scatter(is_val, eisc_val, s=120, label=label)
    ax.set_xlabel("IS ")
    ax.set_ylabel("EISC ")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(config.FIGURES_DIR, "fig7_eisc_vs_is.png")
    if save:
        plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path

def plot_image_grid(real_images_by_class, lstm_images_by_class, transformer_images_by_class, class_names=None, n_cols=5, save=True):
    n_classes = len(real_images_by_class)
    fig_list = []
    for cls_idx in range(n_classes):
        fig, axes = plt.subplots(3, n_cols, figsize=(n_cols * 2, 7))
        for row, (imgs, row_title) in enumerate([(real_images_by_class[cls_idx], "Real"), (lstm_images_by_class[cls_idx], "LSTM"), (transformer_images_by_class[cls_idx], "Transformer")]):
            for col in range(n_cols):
                ax = axes[row][col]
                if col < len(imgs): ax.imshow(imgs[col])
                ax.axis("off")
        plt.tight_layout()
        path = os.path.join(config.FIGURES_DIR, f"fig3_grid_class{cls_idx}.png")
        if save: plt.savefig(path, dpi=100, bbox_inches="tight")
        plt.close()
        fig_list.append(path)
    return fig_list

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fig", choices=["tsne", "training_curves", "ablation", "scatter", "demo_all"], default="demo_all")
    p.add_argument("--emb_transformer", type=str, default="")
    p.add_argument("--emb_lstm",        type=str, default="")
    p.add_argument("--labels",          type=str, default="")
    p.add_argument("--g_losses_t",      type=str, default="")
    p.add_argument("--d_losses_t",      type=str, default="")
    p.add_argument("--g_losses_l",      type=str, default="")
    p.add_argument("--d_losses_l",      type=str, default="")
    p.add_argument("--dataset",         type=str, default="objects")
    return p.parse_args()

def demo_all_figures():
    np.random.seed(config.SEED)
    N, n_classes = 230, 10
    labels = np.repeat(np.arange(n_classes), N // n_classes)
    emb_lstm = np.random.randn(N, 128) * 2.0
    emb_transformer = np.zeros((N, 128))
    for i, lbl in enumerate(labels): emb_transformer[i] = np.random.randn(128) * 0.5 + lbl * 1.2
    plot_tsne(emb_transformer, emb_lstm, labels)
    epochs = 200
    G_t, D_t = np.random.rand(epochs) * 0.5 + np.linspace(3, 1.5, epochs), np.random.rand(epochs) * 0.3 + np.linspace(1.5, 0.8, epochs)
    G_l, D_l = np.random.rand(epochs) * 0.5 + np.linspace(3.5, 2.0, epochs), np.random.rand(epochs) * 0.3 + np.linspace(1.8, 1.0, epochs)
    plot_training_curves([G_t.tolist(), G_l.tolist()], [D_t.tolist(), D_l.tolist()], ["Transformer", "LSTM"])
    plot_ablation_bars(["L1_mean", "L2_mean", "L4_mean", "L2_cls", "L2_noAug"], [6.1, 6.9, 6.5, 6.7, 6.2], [0.41, 0.48, 0.45, 0.46, 0.40], [0.48, 0.58, 0.54, 0.56, 0.50])
    plot_eisc_vs_is([6.78, 7.1, 5.43, 4.93], [0.42, 0.48, 0.35, 0.31], ["LSTM", "Transformer", "ThoughtViz", "AC-GAN"])

if __name__ == "__main__":
    args = parse_args()
    if args.fig == "demo_all": demo_all_figures()
    elif args.fig == "tsne":
        plot_tsne(np.load(args.emb_transformer), np.load(args.emb_lstm), np.load(args.labels))
    elif args.fig == "training_curves":
        G_t, D_t = np.load(args.g_losses_t).tolist(), np.load(args.d_losses_t).tolist()
        if args.g_losses_l and args.d_losses_l:
            plot_training_curves([G_t, np.load(args.g_losses_l).tolist()], [D_t, np.load(args.d_losses_l).tolist()], ["Transformer", "LSTM"], dataset=args.dataset)
        else:
            plot_training_curves([G_t], [D_t], ["Transformer"], dataset=args.dataset)

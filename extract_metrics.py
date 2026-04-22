"""
extract_metrics.py
------------------
Offline K-Means accuracy extraction from saved embeddings.
Does NOT require GPU or CLIP. Run this after a Kaggle training run.

Usage:
    python extract_metrics.py
    python extract_metrics.py --ckpt_dir path/to/checkpoints --dataset imagenet
"""
import argparse
import numpy as np
import os

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", default="results (1)/checkpoints",
                   help="Directory containing embeddings_*.npy and labels_*.npy")
    p.add_argument("--dataset", default="imagenet", choices=["imagenet", "objects", "characters"])
    p.add_argument("--n_seeds", type=int, default=5, help="Number of K-Means seeds to average")
    return p.parse_args()

def hungarian_kmeans_acc(embs, labels, seed=42):
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.preprocessing import normalize
    from scipy.optimize import linear_sum_assignment

    n_clusters    = len(np.unique(labels))
    embs_norm     = normalize(embs)
    km = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed, n_init=10, max_iter=500)
    preds         = km.fit_predict(embs_norm)
    unique_labels = np.unique(labels)
    lbl_to_idx    = {l: i for i, l in enumerate(unique_labels)}
    C = np.zeros((n_clusters, n_clusters), dtype=np.int64)
    for p, l in zip(preds, labels):
        C[p, lbl_to_idx[l]] += 1
    row_ind, col_ind = linear_sum_assignment(-C)
    return C[row_ind, col_ind].sum() / len(labels)

def compute_metrics(embs, labels, seeds):
    accs = [hungarian_kmeans_acc(embs, labels, seed=s)
            for s in range(seeds)]
    return float(np.mean(accs)), float(np.std(accs))

def main():
    args = parse_args()
    ds   = args.dataset

    lbl_path = os.path.join(args.ckpt_dir, f"labels_{ds}.npy")
    emb_t_path = os.path.join(args.ckpt_dir, f"embeddings_transformer_{ds}_main.npy")
    emb_l_path = os.path.join(args.ckpt_dir, f"embeddings_lstm_{ds}_baseline.npy")

    if not os.path.isfile(lbl_path):
        raise FileNotFoundError(f"Labels not found: {lbl_path}")

    labels = np.load(lbl_path)
    print(f"\n{'='*55}")
    print(f" Dataset : {ds}")
    print(f" Samples : {len(labels)}")
    print(f" Classes : {len(np.unique(labels))}")
    print(f" Seeds   : {args.n_seeds}")
    print(f"{'='*55}")

    results = {}

    if os.path.isfile(emb_t_path):
        embs = np.load(emb_t_path)
        print(f"\n[Transformer]  shape={embs.shape}")
        mean, std = compute_metrics(embs, labels, args.n_seeds)
        print(f"  K-Means Acc = {mean:.4f} ± {std:.4f}  ({mean*100:.2f}%)")
        results["transformer"] = {"kmeans_mean": mean, "kmeans_std": std}
    else:
        print(f"[Transformer] embeddings not found: {emb_t_path}")

    if os.path.isfile(emb_l_path):
        embs = np.load(emb_l_path)
        print(f"\n[LSTM]         shape={embs.shape}")
        mean, std = compute_metrics(embs, labels, args.n_seeds)
        print(f"  K-Means Acc = {mean:.4f} ± {std:.4f}  ({mean*100:.2f}%)")
        results["lstm"] = {"kmeans_mean": mean, "kmeans_std": std}
    else:
        print(f"[LSTM] embeddings not found: {emb_l_path}")

    print(f"\n{'='*55}")
    if "transformer" in results and "lstm" in results:
        delta = results["transformer"]["kmeans_mean"] - results["lstm"]["kmeans_mean"]
        print(f" Transformer vs LSTM delta: {delta:+.4f} ({delta*100:+.2f}%)")

    # Print summary table
    print("\n── Results Table ──────────────────────────────")
    print(f"{'Method':<14} {'K-Means':>10} {'±':>6}")
    for name, r in results.items():
        print(f"{name:<14} {r['kmeans_mean']*100:>9.2f}% {r['kmeans_std']*100:>5.2f}%")
    print()

if __name__ == "__main__":
    main()

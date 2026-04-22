import argparse
import numpy as np
import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import config

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", default=config.CHECKPOINT_DIR)
    p.add_argument("--dataset", default="imagenet", choices=["imagenet", "objects", "characters"])
    p.add_argument("--n_seeds", type=int, default=5)
    return p.parse_args()

def hungarian_kmeans_acc(embs, labels, seed=42):
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.preprocessing import normalize
    from scipy.optimize import linear_sum_assignment
    n_clusters = len(np.unique(labels))
    embs_norm = normalize(embs)
    km = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed, n_init=10, max_iter=500)
    preds = km.fit_predict(embs_norm)
    unique_labels = np.unique(labels)
    lbl_to_idx = {l: i for i, l in enumerate(unique_labels)}
    C = np.zeros((n_clusters, n_clusters), dtype=np.int64)
    for p, l in zip(preds, labels):
        C[p, lbl_to_idx[l]] += 1
    row_ind, col_ind = linear_sum_assignment(-C)
    return C[row_ind, col_ind].sum() / len(labels)

def compute_metrics(embs, labels, seeds):
    accs = [hungarian_kmeans_acc(embs, labels, seed=s) for s in range(seeds)]
    return float(np.mean(accs)), float(np.std(accs))

def main():
    args = parse_args()
    ds = args.dataset
    lbl_path = os.path.join(args.ckpt_dir, f"labels_{ds}.npy")
    emb_t_path = os.path.join(args.ckpt_dir, f"embeddings_transformer_{ds}_main.npy")
    emb_l_path = os.path.join(args.ckpt_dir, f"embeddings_lstm_{ds}_baseline.npy")
    if not os.path.isfile(lbl_path):
        raise FileNotFoundError(f"Labels not found: {lbl_path}")
    labels = np.load(lbl_path)
    results = {}
    if os.path.isfile(emb_t_path):
        embs = np.load(emb_t_path)
        mean, std = compute_metrics(embs, labels, args.n_seeds)
        results["transformer"] = {"kmeans_mean": mean, "kmeans_std": std}
    if os.path.isfile(emb_l_path):
        embs = np.load(emb_l_path)
        mean, std = compute_metrics(embs, labels, args.n_seeds)
        results["lstm"] = {"kmeans_mean": mean, "kmeans_std": std}
    if "transformer" in results and "lstm" in results:
        delta = results["transformer"]["kmeans_mean"] - results["lstm"]["kmeans_mean"]
    for name, r in results.items():
        print(f"{name}: {r['kmeans_mean']*100:.2f}% ± {r['kmeans_std']*100:.2f}%")

if __name__ == "__main__":
    main()

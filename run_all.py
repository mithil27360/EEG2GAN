import os
import sys
import argparse
import subprocess
import json
import time
import config

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["objects", "characters", "mindbigdata", "imagenet"], default="objects")
    p.add_argument("--dummy", action="store_true")
    p.add_argument("--enc_epochs", type=int, default=config.ENC_EPOCHS)
    p.add_argument("--gan_epochs", type=int, default=config.GAN_EPOCHS)
    p.add_argument("--skip_baseline", action="store_true")
    p.add_argument("--skip_ablations", action="store_true")
    p.add_argument("--skip_visuals", action="store_true")
    return p.parse_args()

def run(cmd, label=""):
    print(f"\n   {label}")
    ret = subprocess.run(cmd)
    return ret.returncode == 0

def main():
    args = parse_args()
    py = sys.executable
    ds = args.dataset
    dummy_flag = ["--dummy"] if args.dummy else []
    n_gen_eval = getattr(config, 'IS_N_SAMPLES', 2048)
    eval_flags = ["--n_gen", "230"] if args.dummy else ["--n_gen", str(n_gen_eval)]
    
    if not args.skip_baseline:
        run([py, "train_encoder.py", "--encoder", "lstm", "--dataset", ds, "--epochs", str(args.enc_epochs), "--tag", "baseline"] + dummy_flag, label="Exp 1a")
        run([py, "train_gan.py", "--encoder_ckpt", os.path.join(config.CHECKPOINT_DIR, f"encoder_lstm_{ds}_baseline.pth"), "--encoder_type", "lstm", "--dataset", ds, "--epochs", str(args.gan_epochs), "--tag", "baseline", "--resume"] + dummy_flag, label="Exp 1b")
        run([py, "evaluate.py", "--encoder_ckpt", os.path.join(config.CHECKPOINT_DIR, f"encoder_lstm_{ds}_baseline.pth"), "--gan_ckpt", os.path.join(config.CHECKPOINT_DIR, f"gan_lstm_{ds}_baseline.pth"), "--encoder_type", "lstm", "--dataset", ds, "--no_eisc", "--output_csv", "results_main.csv"] + dummy_flag + eval_flags, label="Exp 1c")
        
    run([py, "train_encoder.py", "--encoder", "transformer", "--dataset", ds, "--epochs", str(args.enc_epochs), "--tag", "main"] + dummy_flag, label="Exp 2a")
    run([py, "train_gan.py", "--encoder_ckpt", os.path.join(config.CHECKPOINT_DIR, f"encoder_transformer_{ds}_main.pth"), "--encoder_type", "transformer", "--dataset", ds, "--epochs", str(args.gan_epochs), "--tag", "main", "--resume"] + dummy_flag, label="Exp 2b")
    run([py, "evaluate.py", "--encoder_ckpt", os.path.join(config.CHECKPOINT_DIR, f"encoder_transformer_{ds}_main.pth"), "--gan_ckpt", os.path.join(config.CHECKPOINT_DIR, f"gan_transformer_{ds}_main.pth"), "--encoder_type", "transformer", "--dataset", ds, "--output_csv", "results_main.csv"] + dummy_flag + eval_flags, label="Exp 2c")
    
    if not args.skip_ablations:
        run([py, "ablation_study.py", "--dataset", ds, "--enc_epochs", str(args.enc_epochs), "--gan_epochs", str(args.gan_epochs), "--output_csv", "results_ablation.csv"] + dummy_flag, label="Exp 3")
    if not args.skip_visuals:
        run([py, "visualize.py", "--fig", "demo_all"], label="Figures")
        emb_t = os.path.join(config.CHECKPOINT_DIR, f"embeddings_transformer_{ds}_main.npy")
        emb_l = os.path.join(config.CHECKPOINT_DIR, f"embeddings_lstm_{ds}_baseline.npy")
        lbl = os.path.join(config.CHECKPOINT_DIR, f"labels_{ds}.npy")
        if os.path.isfile(emb_t) and os.path.isfile(emb_l) and os.path.isfile(lbl):
            run([py, "visualize.py", "--fig", "tsne", "--emb_transformer", emb_t, "--emb_lstm", emb_l, "--labels", lbl], label="Fig 2")
        G_t = os.path.join(config.OUTPUT_DIR, f"G_losses_transformer_{ds}_main.npy")
        D_t = os.path.join(config.OUTPUT_DIR, f"D_losses_transformer_{ds}_main.npy")
        G_l = os.path.join(config.OUTPUT_DIR, f"G_losses_lstm_{ds}_baseline.npy")
        D_l = os.path.join(config.OUTPUT_DIR, f"D_losses_lstm_{ds}_baseline.npy")
        if os.path.isfile(G_t) and os.path.isfile(D_t):
            run([py, "visualize.py", "--fig", "training_curves", "--g_losses_t", G_t, "--d_losses_t", D_t, "--g_losses_l", G_l, "--d_losses_l", D_l, "--dataset", ds], label="Fig 5")

if __name__ == "__main__":
    main()

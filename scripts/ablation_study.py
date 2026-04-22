import os
import sys
import argparse
import subprocess
import csv
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import config

ABLATION_CONFIGS = [
    ("L1_mean_aug", ["--n_layers", "1", "--pooling", "mean"], []),
    ("L2_mean_aug", ["--n_layers", "2", "--pooling", "mean"], []),
    ("L4_mean_aug", ["--n_layers", "4", "--pooling", "mean"], []),
    ("L2_cls_aug", ["--n_layers", "2", "--pooling", "cls"], []),
    ("L2_mean_noaug", ["--n_layers", "2", "--pooling", "mean"], ["--no_diffaug"]),
]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["objects", "characters", "mindbigdata", "imagenet"], default="objects")
    p.add_argument("--dummy", action="store_true")
    p.add_argument("--enc_epochs", type=int, default=500)
    p.add_argument("--gan_epochs", type=int, default=200)
    p.add_argument("--batch_enc", type=int, default=32)
    p.add_argument("--batch_gan", type=int, default=16)
    p.add_argument("--output_csv", type=str, default="ablation_results.csv")
    return p.parse_args()

def run_cmd(cmd):
    result = subprocess.run(cmd, capture_output=False, text=True)
    return result.returncode == 0

def main():
    args = parse_args()
    py = sys.executable
    for tag, enc_extra, gan_extra in ABLATION_CONFIGS:
        enc_cmd = [
            py, "train_encoder.py",
            "--encoder", "transformer",
            "--dataset", args.dataset,
            "--epochs", str(args.enc_epochs),
            "--batch_size", str(args.batch_enc),
            "--tag", tag,
        ]
        if args.dummy:
            enc_cmd.append("--dummy")
        enc_cmd += enc_extra
        if not run_cmd(enc_cmd):
            continue
        enc_ckpt = os.path.join(config.CHECKPOINT_DIR, f"encoder_transformer_{args.dataset}_{tag}.pth")
        gan_cmd = [
            py, "train_gan.py",
            "--encoder_ckpt", enc_ckpt,
            "--encoder_type", "transformer",
            "--dataset", args.dataset,
            "--epochs", str(args.gan_epochs),
            "--batch_size", str(args.batch_gan),
            "--tag", tag,
        ]
        if args.dummy:
            gan_cmd.append("--dummy")
        gan_cmd += gan_extra
        if not run_cmd(gan_cmd):
            continue
        gan_ckpt = os.path.join(config.CHECKPOINT_DIR, f"gan_transformer_{args.dataset}_{tag}.pth")
        eval_cmd = [
            py, "evaluate.py",
            "--encoder_ckpt", enc_ckpt,
            "--gan_ckpt", gan_ckpt,
            "--encoder_type", "transformer",
            "--dataset", args.dataset,
            "--output_csv", args.output_csv,
        ]
        if args.dummy:
            eval_cmd.append("--dummy")
        run_cmd(eval_cmd)

if __name__ == "__main__":
    main()

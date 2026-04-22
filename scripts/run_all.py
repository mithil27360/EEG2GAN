import os
import sys
import argparse
import subprocess
import json
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
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
    # Ensure we run scripts from the scripts directory
    ret = subprocess.run(cmd)
    return ret.returncode == 0

def main():
    args = parse_args()
    py = sys.executable
    ds = args.dataset
    dummy_flag = ["--dummy"] if args.dummy else []
    n_gen_eval = getattr(config, 'IS_N_SAMPLES', 2048)
    eval_flags = ["--n_gen", "230"] if args.dummy else ["--n_gen", str(n_gen_eval)]
    
    # Get absolute paths to scripts
    base_dir = os.path.dirname(os.path.abspath(__file__))
    viz_dir = os.path.join(base_dir, "..", "visualizations")
    
    script_enc = os.path.join(base_dir, "train_encoder.py")
    script_gan = os.path.join(base_dir, "train_gan.py")
    script_eval = os.path.join(base_dir, "evaluate.py")
    script_abl = os.path.join(base_dir, "ablation_study.py")
    
    if not args.skip_baseline:
        run([py, script_enc, "--encoder", "lstm", "--dataset", ds, "--epochs", str(args.enc_epochs), "--tag", "baseline"] + dummy_flag, label="Exp 1a")
        run([py, script_gan, "--encoder_ckpt", os.path.join(config.CHECKPOINT_DIR, f"encoder_lstm_{ds}_baseline.pth"), "--encoder_type", "lstm", "--dataset", ds, "--epochs", str(args.gan_epochs), "--tag", "baseline", "--resume"] + dummy_flag, label="Exp 1b")
        run([py, script_eval, "--encoder_ckpt", os.path.join(config.CHECKPOINT_DIR, f"encoder_lstm_{ds}_baseline.pth"), "--gan_ckpt", os.path.join(config.CHECKPOINT_DIR, f"gan_lstm_{ds}_baseline.pth"), "--encoder_type", "lstm", "--dataset", ds, "--no_eisc", "--output_csv", "results_main.csv"] + dummy_flag + eval_flags, label="Exp 1c")
        
    run([py, script_enc, "--encoder", "transformer", "--dataset", ds, "--epochs", str(args.enc_epochs), "--tag", "main"] + dummy_flag, label="Exp 2a")
    run([py, script_gan, "--encoder_ckpt", os.path.join(config.CHECKPOINT_DIR, f"encoder_transformer_{ds}_main.pth"), "--encoder_type", "transformer", "--dataset", ds, "--epochs", str(args.gan_epochs), "--tag", "main", "--resume"] + dummy_flag, label="Exp 2b")
    run([py, script_eval, "--encoder_ckpt", os.path.join(config.CHECKPOINT_DIR, f"encoder_transformer_{ds}_main.pth"), "--gan_ckpt", os.path.join(config.CHECKPOINT_DIR, f"gan_transformer_{ds}_main.pth"), "--encoder_type", "transformer", "--dataset", ds, "--output_csv", "results_main.csv"] + dummy_flag + eval_flags, label="Exp 2c")
    
    if not args.skip_ablations:
        run([py, script_abl, "--dataset", ds, "--enc_epochs", str(args.enc_epochs), "--gan_epochs", str(args.gan_epochs), "--output_csv", "results_ablation.csv"] + dummy_flag, label="Exp 3")
    
    if not args.skip_visuals:
        # Using the new publication visualization scripts
        run([py, os.path.join(viz_dir, "paper_figs_part1.py")], label="Main Figures")
        run([py, os.path.join(viz_dir, "paper_figs_part2.py")], label="Supplementary 1")
        run([py, os.path.join(viz_dir, "paper_figs_part3.py")], label="Supplementary 2")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()

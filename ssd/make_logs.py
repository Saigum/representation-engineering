import os
import subprocess
import glob
import sys
import argparse
import yaml, os, itertools, re
import argparse 
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--log_dir', type=str, default='logs', help='Directory to store log files')
parser.add_argument('--concept', type=str, choices=["anger","happiness","honesty","disgust","sadness"],default="anger", help='Available concepts for the sweep')
parser.add_argument('--selection_low', type=float, default=0.0, help='Lower bound for selection weighting')
parser.add_argument('--selection_high', type=float, default=10000.0, help='Upper bound for selection weighting')
parser.add_argument('--dampen_low', type=float, default=0.0, help='Lower bound for dampening constant')
parser.add_argument('--dampen_high', type=float, default=1.0, help='Upper bound for dampening constant')
parser.add_argument('--num_points', type=int, default=4, help='Number of points to sample in each dimension')

args = parser.parse_args()

BASE = "configs/ssd_config.yaml"  # your base YAML
OUTDIR = f"configs/sweep/{args.concept}/"
CHECKPOINT_DIR = f"checkpoints/{args.concept}/"
SAVE_DIR = f"save_dir/{args.concept}/"

os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)
selection_vals = np.linspace(args.selection_low, args.selection_high, args.num_points)
dampen_vals = np.linspace(args.dampen_low, args.dampen_high, args.num_points)
def ftag(x: float) -> str:
    """Safe tag for filenames: 0.1 -> 0p1, 1e-3 -> 1e-3"""
    s = f"{x}".replace(".", "p")
    # keep scientific notation intact if present
    s = re.sub(r"(?<=\d)\.(?=\d)", "p", s)
    return s

base = yaml.safe_load(open(BASE))

for sel, damp in itertools.product(selection_vals, dampen_vals):
    cfg = yaml.safe_load(open(BASE))  # deep-ish copy by reload
    # ensure param_config exists
    cfg.setdefault("param_config", {})
    cfg["param_config"]["selection_weighting"] = float(sel)
    cfg["param_config"]["dampening_constant"]  = float(damp)
    cfg["data_args"]["concept"] = args.concept

    # Make save_path unique so checkpoints don’t overwrite
    sel_t, damp_t = ftag(sel), ftag(damp)
    tag = f"sel_{sel_t}_damp_{damp_t}"
    cfg.setdefault("model_args", {})
    
    cfg["model_args"]["save_path"] = f"checkpoints/{args.concept}/{tag}.pth"
    cfg["model_args"]["save_dir"] = f"save_dir/{args.concept}/{tag}/"
    # Write the config
    out = os.path.join(OUTDIR, f"{tag}.yaml")
    with open(out, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


print("Created all config files .....")


os.makedirs('logs', exist_ok=True)

# Get command line argument for config directory
config_dir = f"configs/sweep/{args.concept}/"

# Process each config file
for cfg in glob.glob(f"{config_dir}/*.yaml"):
  tag = os.path.splitext(os.path.basename(cfg))[0]
  print(f">>> {tag}")
  
  # Run training
  with open(f"logs/{tag}.log", 'w') as f:
    subprocess.run(['python', 'unlearning_.py', '--config', cfg], 
            stdout=f, stderr=subprocess.STDOUT)
  
  # Run inference
  with open(f"logs/{tag}_inference.log", 'w') as f:
    subprocess.run(['python', 'run_inference.py', '--config', cfg],
            stdout=f, stderr=subprocess.STDOUT)
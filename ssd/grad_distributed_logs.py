import os
import subprocess
import glob
import sys
import argparse
import yaml, os, itertools, re
import argparse 
import numpy as np
import multiprocessing # <-- Import multiprocessing

# --- Worker Function ---
# grad_distributed_logs.py (drop-in replacement for the parallel section)
import math

def worker_for_gpu(gpu_id:int, cfg_files:list[str], log_dir:str):
    # Pin this worker to one GPU for all its subprocesses
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    # (optional) avoid CPU thrash when you have many procs
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")

    results = []
    for config_file in cfg_files:
        tag = os.path.splitext(os.path.basename(config_file))[0]
        print(f">>> [GPU {gpu_id}] Starting job: {tag}")
        try:
            with open(f"{log_dir}/{tag}.log", 'w') as f:
                subprocess.run(
                    ['python', 'grad_parameterperturber.py', '--config', config_file],
                    stdout=f, stderr=subprocess.STDOUT, env=env, check=True
                )
            with open(f"{log_dir}/{tag}_inference.log", 'w') as f:
                subprocess.run(
                    ['python', 'run_inference.py', '--config', config_file],
                    stdout=f, stderr=subprocess.STDOUT, env=env, check=True
                )
            with open(f"{log_dir}/{tag}_sentiment_eval.log", 'w') as f:
                subprocess.run(
                    ['python', 'classifier_eval.py', '--config', config_file],
                    stdout=f, stderr=subprocess.STDOUT, env=env, check=True
                )
            print(f"<<< [GPU {gpu_id}] Finished job: {tag}")
            results.append((config_file, "Success"))
        except subprocess.CalledProcessError as e:
            print(f"!!! [GPU {gpu_id}] ERROR on job: {tag}. Check logs.")
            results.append((config_file, f"Failed with code {e.returncode}"))
        except Exception as e:
            print(f"!!! [GPU {gpu_id}] PYTHON ERROR on job: {tag}: {e}")
            results.append((config_file, f"Failed with exception: {e}"))
    return results
    
def ftag(x: float) -> str:
    """Safe tag for filenames: 0.1 -> 0p1, 1e-3 -> 1e-3"""
    s = f"{x}".replace(".", "p")
    # keep scientific notation intact if present
    s = re.sub(r"(?<=\d)\.(?=\d)", "p", s)
    return s

# --- Main execution ---
# CRITICAL: Wrap main code in if __name__ == "__main__":
# This is required for multiprocessing to work correctly

DATASET_PATHS = {
    "honesty": "data/facts/facts_true_false.csv",
    "anger": "data/emotions/anger_train.json",
    "happiness": "data/emotions/happiness.json",
    "sadness": "data/emotions/sadness.json",
    "surprise": "data/emotions/surprise.json",
    "disgust": "data/emotions/disgust.json"
}
if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True) # Good practice for CUDA

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory to store log files')
    parser.add_argument('--concept', type=str, choices=["anger","happiness","honesty","disgust","sadness"],default="anger", help='Available concepts for the sweep')
    parser.add_argument('--selection_low', type=float, default=0.0, help='Lower bound for selection weighting')
    parser.add_argument('--selection_high', type=float, default=10000.0, help='Upper bound for selection weighting')
    parser.add_argument('--dampen_low', type=float, default=0.0, help='Lower bound for dampening constant')
    parser.add_argument('--dampen_high', type=float, default=1.0, help='Upper bound for dampening constant')
    parser.add_argument('--num_points', type=int, default=4, help='Number of points to sample in each dimension')
    
    # --- NEW ARGUMENT ---
    parser.add_argument('--num_gpus', type=int, default=4, help='Number of GPUs to run in parallel')

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

    base = yaml.safe_load(open(BASE))

    for sel, damp in itertools.product(selection_vals, dampen_vals):
        cfg = yaml.safe_load(open(BASE))  # deep-ish copy by reload
        # ensure param_config exists
        cfg.setdefault("param_config", {})
        cfg["param_config"]["selection_weighting"] = float(sel)
        cfg["param_config"]["dampening_constant"]  = float(damp)
        cfg["data_args"]["concept"] = args.concept
        cfg["data_args"]["data_path"] = DATASET_PATHS[args.concept]
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

    os.makedirs(args.log_dir, exist_ok=True) # Use arg.log_dir

    # Get command line argument for config directory
    config_dir = f"configs/sweep/{args.concept}/"

    # --- REPLACED PARALLEL EXECUTION ---
    
    # 1. Get all config files
    config_files = sorted(glob.glob(f"{config_dir}/*.yaml"))
    
    if not config_files:
        print(f"Warning: No config files found in {config_dir}")
        sys.exit(0)
    
    print(f"Found {len(config_files)} configs. Starting parallel run on {args.num_gpus} GPUs.")
    print(f"Found {len(config_files)} configs. Starting pinned run on {args.num_gpus} GPUs.")
    
    # shard configs by GPU: gpu k gets files at indices k, k+G, k+2G, ...
    per_gpu_cfgs = [config_files[k::args.num_gpus] for k in range(args.num_gpus)]
    
    # spawn exactly one process per GPU
    with multiprocessing.Pool(processes=args.num_gpus) as pool:
        per_gpu_results = pool.starmap(worker_for_gpu,
                                       [(k, per_gpu_cfgs[k], args.log_dir) for k in range(args.num_gpus)])
    
    # flatten results
    results = [item for sub in per_gpu_results for item in sub]
    
    print("\n--- Sweep Summary ---")
    success_count = sum(1 for _, status in results if status == "Success")
    for config, status in results:
        print(f"{config}: {status}")
    print(f"\nCompleted: {success_count} / {len(results)} jobs successful.")
    
#     
#     # 2. Create a list of tasks: (config_file, gpu_id)
#     # This cycles GPU IDs: 0, 1, 2, 3, 0, 1, 2, 3, ...
#     tasks = []
#     for i, cfg_file in enumerate(config_files):
#         gpu_id = i % args.num_gpus
#         tasks.append((cfg_file, gpu_id, args.log_dir))
#     
#     # 3. Run the tasks in a parallel pool
#     # 'processes' controls how many jobs run at once
#     with multiprocessing.Pool(processes=args.num_gpus) as pool:
#         # 'starmap' is used to pass multiple arguments (config_file, gpu_id) to run_job
#         results = pool.starmap(run_job, tasks)
# 
#     print("\n--- Sweep Summary ---")
#     success_count = 0
#     for config, status in results:
#         print(f"{config}: {status}")
#         if status == "Success":
#             success_count += 1
#             
#     print(f"\nCompleted: {success_count} / {len(results)} jobs successful.")
import yaml, os, itertools, re

BASE = "configs/ssd_config.yaml"  # your base YAML
OUTDIR = "configs/sweep"
os.makedirs(OUTDIR, exist_ok=True)

# Grids to sweep
selection_vals = [7000, 8000, 9000, 10000]
dampen_vals    = [0.05, 0.1, 0.2, 0.4]

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

    # Make save_path unique so checkpoints don’t overwrite
    sel_t, damp_t = ftag(sel), ftag(damp)
    tag = f"sel_{sel_t}_damp_{damp_t}"
    cfg.setdefault("model_args", {})
    cfg["model_args"]["save_path"] = f"checkpoints/{tag}.pth"
    cfg["model_args"]["save_dir"] = f"save_dir/{tag}/"
    # Write the config
    out = os.path.join(OUTDIR, f"{tag}.yaml")
    with open(out, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print("wrote", out)

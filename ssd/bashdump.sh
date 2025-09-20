# For single-GPU
python unlearn.py --config config.yaml

# For multi-GPU (e.g., 4 GPUs)
torchrun --nproc_per_node=4 unlearn.py --config config.yaml

# Use config but override the model and save path
torchrun --nproc_per_node=4 unlearn.py --config config.yaml --model_name "gpt2-medium" --save_path "gpt2-medium-unlearned.pth"
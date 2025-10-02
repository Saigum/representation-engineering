#!/bin/bash
#!/bin/bash
#SBATCH -w node05
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=4096
#SBATCH --time=96:00:00
#SBATCH --mincpus=4
#SBATCH --mail-user=shivani.kulkarni@students.iiit.ac.in
#SBATCH --mail-type=ALL


# ds_master_port=$((29000 + RANDOM % 1000))


echo "Activating virtualenv"
source ~/ANLP_A1/env/bin/activate


python ~/spk2/anlpproject/llama2_lorra.py \
    --model_name_or_path  "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
    --user_tag '[INST]' \
    --assistant_tag '[/INST]' \
    --pos_type 'a happy' \
    --neg_type 'an neutral' \
    --control_template "Give {type} answer." \
    --target_layers "10,12,14,16,18,20" \
    --lorra_alpha -1 \
    --lorra_beta 0 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --output_dir ./lorra_tqa_1b \
    --overwrite_output_dir \
    --bf16 True \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --save_total_limit 0 \
    --learning_rate 3e-4 \
    --weight_decay 0. \
    --lr_scheduler_type "constant" \
    --logging_strategy "steps" \
    --logging_steps 10 \
    --model_max_length 128 \
    --q_lora False \
    --gradient_checkpointing True \
    --report_to none


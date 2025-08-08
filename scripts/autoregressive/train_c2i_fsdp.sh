# !/bin/bash
set -x
export PYTHONPATH="${PWD}:${PYTHONPATH}"
export WANDB_NAME="run_$(date +%Y%m%d_%H%M%S)"
export WANDB_PROJECT="c2i_selftok"
export WANDB_API_KEY="0b30f581d65172381c1f1a45f928210cab80f1de"

torchrun --nnodes=1 --nproc_per_node=8 --node_rank=0 \
    --master_port=12335 \
    autoregressive/train/train_c2i_fsdp.py \
    --code-path /home/jovyan/datasets/code-imagenet-E31 \
    --gpt-resume /home/jovyan/llamagen-finetune/results_ckpt_e31_rope/20250807071554-GPT-XL/0110000 \
    --cloud-save-path ./results_ckpt_e31_rope \
    --gpt-model GPT-XL --gpt-type c2i \
    --vocab-size 16384 \
    --image-size 512 \
    --global-batch-size 128 \
    --no-local-save 

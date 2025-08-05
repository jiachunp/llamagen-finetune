# !/bin/bash
set -x

export PYTHONPATH="${PWD}:${PYTHONPATH}"
export WANDB_NAME="run_$(date +%Y%m%d_%H%M%S)"
export WANDB_PROJECT="c2i_selftok"
export WANDB_API_KEY="0b30f581d65172381c1f1a45f928210cab80f1de"


torchrun --nnodes=1 --nproc_per_node=8 --node_rank=0 \
    --master_port=20045 \
    autoregressive/train/train_c2i.py \
    --code-path /home/jovyan/datasets/code-imagenet \
    --cloud-save-path ./results_ckpt \
    --gpt-model GPT-L --gpt-type c2i \
    --vocab-size 32768 \
    --image-size 256 \
    --global-batch-size 512 \
    --no-local-save 
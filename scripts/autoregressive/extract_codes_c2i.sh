# !/bin/bash
set -x

torchrun \
    --nnodes=1 --nproc_per_node=8 --node_rank=0 \
    --master_port=12335 \
    autoregressive/train/extract_codes_c2i.py \
    --yml-path /home/jovyan/SelftokTokenizer/configs/renderer/512.yml \
    --vq-ckpt /home/jovyan/ckpt/selftok-e31/E31_renderer.safetensors \
    --sd3-pretrained /home/jovyan/ckpt/stable_diffusion_3_medium/sd3_medium.safetensors \
    --data-path /home/jovyan/datasets/imagenet-1k-vl-enriched/data \
    --code-path /home/jovyan/datasets/code-imagenet-E31 \
    --image-size 512



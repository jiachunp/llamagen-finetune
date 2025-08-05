# !/bin/bash
set -x

torchrun \
    --nnodes=1 --nproc_per_node=8 --node_rank=0 \
    --master_port=12335 \
    autoregressive/train/extract_codes_c2i.py \
    --yml-path /home/jovyan/SelftokTokenizer-cnt/configs/res256/256-eval.yml \
    --vq-ckpt /home/jovyan/ckpt/selftok/tokenizer_512_ckpt.pth \
    --sd3-pretrained /home/jovyan/ckpt/sd3/sd3_medium.safetensors \
    --data-path /home/jovyan/datasets/imagenet/data \
    --code-path /home/jovyan/datasets/code-imagenet \
    --image-size 256 



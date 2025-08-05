# # !/bin/bash
# set -x

# torchrun \
# --nnodes=1 --nproc_per_node=8 --node_rank=0 \
# --master_port=12345 \
# autoregressive/sample/sample_c2i_ddp.py \
# --vq-ckpt ./pretrained_models/vq_ds16_c2i.pt \
# "$@"

# # !/bin/bash
# set -x

torchrun \
    --nnodes=1 --nproc_per_node=1 --node_rank=0 \
    --master_port=12345 \
    autoregressive/sample/sample_c2i.py \
    --yml-path="/home/jovyan/SelftokTokenizer-cnt/configs/renderer/renderer-eval.yml" \
    --vq-ckpt="/home/jovyan/ckpt/selftok/renderer_512_ckpt.pth" \
    --gpt-model="GPT-L" \
    --gpt-ckpt="/home/jovyan/LlamaGen/results_ckpt/2025-08-04-09-41-01/000-GPT-L/checkpoints/0380000.pt" \
    --sd3-pretrained="/home/jovyan/ckpt/sd3/sd3_medium.safetensors" \
    --gpt-type="c2i" \
    --codebook-size=32768 \
    --image-size=256 \
    --cfg-scale 3.0 \
	--top-p 1.0 \
	--top-k 0 \
	--temperature 1.0 \
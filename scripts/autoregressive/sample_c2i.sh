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
    --yml-path="/home/jovyan/SelftokTokenizer/configs/renderer/512.yml" \
    --vq-ckpt="/home/jovyan/ckpt/selftok-e31/E31_renderer.safetensors" \
    --gpt-model="GPT-XL" \
    --gpt-ckpt="/home/jovyan/llamagen-finetune/results_ckpt_e31/20250805151125-GPT-XL/0110000/consolidated.pth" \
    --sd3-pretrained="/home/jovyan/ckpt/stable_diffusion_3_medium/sd3_medium.safetensors" \
    --gpt-type="c2i" \
    --codebook-size=16384 \
    --image-size=512 \
    --cfg-scale 3.0 \
	--top-p 1.0 \
	--top-k 0 \
	--temperature 1.0 \
    --from-fsdp
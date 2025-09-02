cd generative-models

# add generatie-models to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)

CUDA_VISIBLE_DEVICES=5 python scripts/generate_svd_maps.py \
--dataset=animal_kingdom \
--num_frames=25 \
--feature_stride=4 \
--offset_frames=0 \
--diffusion_step=20 \
--start_idx=18750 \
--end_idx=22500 \
--version sd_v2 \
--exp_name=extract_animal_kingdom_sdv2 \

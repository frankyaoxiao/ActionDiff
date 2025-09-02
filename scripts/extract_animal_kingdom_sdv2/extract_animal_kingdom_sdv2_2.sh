cd generative-models

# add generatie-models to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)

CUDA_VISIBLE_DEVICES=1 python scripts/generate_svd_maps.py \
--dataset=animal_kingdom \
--num_frames=25 \
--feature_stride=4 \
--offset_frames=0 \
--diffusion_step=20 \
--start_idx=3750 \
--end_idx=7500 \
--version sd_v2 \
--exp_name=extract_animal_kingdom_sdv2 \

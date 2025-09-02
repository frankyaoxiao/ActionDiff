cd generative-models

# add generatie-models to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)

CUDA_VISIBLE_DEVICES=3 python scripts/generate_svd_maps.py \
--dataset=charades \
--num_frames=25 \
--feature_stride=4 \
--offset_frames=0 \
--diffusion_step=20 \
--start_idx=3000 \
--end_idx=4000 \
--version=sd_v2 \
--exp_name=extract_charades_ego_sdv2 \

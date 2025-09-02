cd generative-models

# add generatie-models to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)

CUDA_VISIBLE_DEVICES=0 python scripts/generate_svd_maps.py \
--dataset=charades \
--num_frames=25 \
--feature_stride=4 \
--offset_frames=0 \
--diffusion_step=10 \
--version svd_xt \
--exp_name=extract_charades_ego_svd_xt_grid_search_t=10_3

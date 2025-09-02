cd generative-models

# add generatie-models to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)

CUDA_VISIBLE_DEVICES=3 python scripts/generate_svd_maps.py \
--dataset=hmdb \
--num_frames=25 \
--feature_stride=4 \
--offset_frames=0 \
--diffusion_step=15 \
--version svd_xt \
--exp_name=extract_hmdb_svd_xt_grid_search_t=15_4

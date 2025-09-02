cd generative-models

# add generatie-models to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)

CUDA_VISIBLE_DEVICES=2 python scripts/generate_svd_maps.py \
--dataset=animal_kingdom \
--num_frames=25 \
--feature_stride=4 \
--offset_frames=0 \
--diffusion_step=10 \
--version svd_xt \
--exp_name=extract_animal_kingdom_svd_xt_grid_search_t=10_3

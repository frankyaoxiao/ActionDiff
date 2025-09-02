cd generative-models

# add generatie-models to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)

CUDA_VISIBLE_DEVICES=1 python scripts/generate_svd_maps.py \
--dataset=animal_kingdom \
--num_frames=25 \
--feature_stride=4 \
--offset_frames=0 \
--diffusion_step=20 \
--version svd_xt \
--exp_name=extract_animal_kingdom_25_frames_svd-xt \
--shuffle=True \
--show_time=True \

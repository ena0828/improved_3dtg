cd ../examples
export CUDA_VISIBLE_DEVICES=$1
results_dir="../results/textured_gaussians_rgba"
python simple_trainer_textured_gaussians.py mcmc \
    --data_dir "datasets/nerf_synthetic/nerf_synthetic/lego/" \
    --pretrained_path "../results/2dgs/lego/ckpts/ckpt_29999.pt" \
    --result_dir "../results/textured_gaussians_rgba/lego" \
    --dataset "blender" \
    --init_extent 1 \
    --init_type "pretrained" \
    --background_mode "white" \
    --model_type=textured_gaussians \
    --init_num_pts=10000 \
    --strategy.cap-max=10000 \
    --strategy.refine-start-iter=1000000000000 \
    --alpha_loss \
    --textured_rgb \
    --textured_alpha \
    --port 6070 \
    > run2_log.txt 2>&1
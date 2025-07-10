cd ../examples
export CUDA_VISIBLE_DEVICES=$1
results_dir="../results/2dgs"
python simple_trainer_textured_gaussians.py mcmc \
    --data_dir "datasets/nerf_synthetic/nerf_synthetic/lego" \
    --result_dir "../results/2dgs/lego" \
    --dataset "blender" \
    --init_extent 1 \
    --init_type "random" \
    --background_mode "white" \
    --model_type=2dgs \
    --init_num_pts=10000 \
    --strategy.cap-max=10000 \
    --alpha_loss \
    --port 6070 \
    > run_log.txt 2>&1

python simple_trainer_textured_gaussians.py mcmc \
    --data_dir "datasets/nerf_synthetic/nerf_synthetic/lego" \
    --result_dir "../results/2dgs/lego" \
    --dataset "blender" \
    --init_extent 1 \
    --init_type "random" \
    --background_mode "white" \
    --model_type=2dgs \
    --init_num_pts=10000 \
    --strategy.cap-max=10000 \
    --alpha_loss \
    --port 6070 
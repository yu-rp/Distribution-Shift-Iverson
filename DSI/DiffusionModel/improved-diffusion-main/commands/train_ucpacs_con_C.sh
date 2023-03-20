cd ~/Repos/DiffusionAttributionCompelete_mulgpu/DiffusionModel/improved-diffusion-main

export OPENAI_LOGDIR="~/Repos/DiffusionAttributionCompelete_mulgpu/DiffusionModelFineTune/cartoon_con"

MODEL_FLAGS="--image_size 64 --num_channels 192 --num_res_blocks 3 --learn_sigma True --class_cond True"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine --rescale_learned_sigmas False --rescale_timesteps False"
TRAIN_FLAGS="--lr 3e-4 --batch_size 16 --resume_checkpoint ~/Repos/DiffusionAttributionCompelete_mulgpu/DiffusionModel/improved-diffusion-main/checkpoints/imagenet64_cond_270M_250K.pt "
DATA_PATH="~/Data/PACSm/cartoon"
TRASH_PATH="~/Repos/DiffusionAttributionCompelete_mulgpu/DiffusionModelFineTune/cartoon_con/trash"

nohup python -m torch.distributed.launch --nproc_per_node=2 --master_port 12353 --nnodes=1 scripts/image_train.py --data_dir $DATA_PATH $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS > $TRASH_PATH 2>&1 &
# python -m torch.distributed.launch --nproc_per_node=1 --master_port 12353 --nnodes=1 scripts/image_train.py --data_dir $DATA_PATH $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS 
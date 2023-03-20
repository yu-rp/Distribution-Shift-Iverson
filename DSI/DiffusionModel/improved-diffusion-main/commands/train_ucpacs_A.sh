cd ~/Repos/DiffusionAttributionCompelete_mulgpu/DiffusionModel/improved-diffusion-main

export OPENAI_LOGDIR="~/Repos/DiffusionAttributionCompelete_mulgpu/DiffusionModelFineTune/art_painting"

MODEL_FLAGS="--image_size 64 --num_channels 128 --num_res_blocks 3 --learn_sigma True "
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine "
TRAIN_FLAGS="--lr 1e-4 --batch_size 32 --resume_checkpoint ~/Repos/DiffusionAttributionCompelete_mulgpu/DiffusionModelFineTune/art_painting/model480000.pt "
DATA_PATH="~/Data/PACSm/art_painting"
TRASH_PATH="~/Repos/DiffusionAttributionCompelete_mulgpu/DiffusionModelFineTune/art_painting/trash"

nohup python -m torch.distributed.launch --nproc_per_node=2 --master_port 12353 --nnodes=1 scripts/image_train.py --data_dir $DATA_PATH $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS > $TRASH_PATH 2>&1 &
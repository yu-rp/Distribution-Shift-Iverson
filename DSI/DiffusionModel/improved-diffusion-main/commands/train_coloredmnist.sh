cd ~/Repos/DiffusionAttributionCompelete_mulgpu/DiffusionModel/improved-diffusion-main

export OPENAI_LOGDIR="~/Repos/DiffusionAttributionCompelete_mulgpu/DiffusionModelFineTune/ColoredMNIST2"

MODEL_FLAGS="--image_size 32 --num_channels 128 --num_res_blocks 3 --learn_sigma True --dropout 0.3"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine"
# TRAIN_FLAGS="--lr 1e-4 --batch_size 128 --resume_checkpoint ~/Repos/DiffusionAttributionCompelete_mulgpu/DiffusionModel/improved-diffusion-main/checkpoints/cifar10_uncond_50M_500K.pt "
TRAIN_FLAGS="--lr 1e-4 --batch_size 128 --resume_checkpoint ~/Repos/DiffusionAttributionCompelete_mulgpu/DiffusionModelFineTune/ColoredMNIST/model050000.pt "
DATA_PATH="ColoredMNIST "
TRASH_PATH="~/Repos/DiffusionAttributionCompelete_mulgpu/DiffusionModelFineTune/ColoredMNIST2/trash"

nohup python -m torch.distributed.launch --nproc_per_node=2 --master_port 12353 --nnodes=1 scripts/image_train.py --data_dir $DATA_PATH $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS > $TRASH_PATH 2>&1 &
"""
Train a diffusion model on images.
"""

import argparse
import logging
import re
import torch
import torch.cuda as cuda
import torch.distributed as distributed

# from improved_diffusion import dist_util, logger
from improved_diffusion import logger
from improved_diffusion.image_datasets import load_data
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from improved_diffusion.train_util import TrainLoop

def get_resume_step(resume_checkpoint):
    if resume_checkpoint is None or resume_checkpoint == "":
        return 0
    else:
        match = re.search("model(\d{6})\.pt", resume_checkpoint)
        if match is None:
            return 0
        else:
            return int(match[1])

def main():
    args = create_argparser().parse_args() 

    local_rank = int(args.local_rank)
    print(local_rank)
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    print(device)
    cuda.set_device(local_rank)
    distributed.init_process_group("nccl",init_method='env://')

    # dist_util.setup_dist() # 配置 多进程
    logger.configure()# logger 句柄

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion( # 建立模型结构
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(device) # 分配模型到 不同gpu
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion) # 建立 schedule sampler

    print(args.data_dir)

    if args.data_dir.startswith("pacs:"):
        datapaths = []
        pacs_paths = {
            "a":"~/Data/PACSm/art_painting",
            "c":"~/Data/PACSm/cartoon",
            "p":"~/Data/PACSm/photo",
            "s":"~/Data/PACSm/sketch",
        }
        for domain in args.data_dir[5:].split(","):
            datapaths.append(pacs_paths[domain])
    elif args.data_dir == "ColoredMNIST":
        print("use colored MNIST")
        datapaths = [
            "~/Data/ColoredMNIST/1_train",
            "~/Data/ColoredMNIST/2_train",
            "~/Data/ColoredMNIST/3_test",
            ]
    elif "cdsprites" in args.data_dir:
        print("use colored cdsprites")
        datapaths = [
            "~/Data/cdsprites",
            int(args.data_dir[9:])
            ]
    else:
        datapaths = [args.data_dir]

    logger.log("creating data loader...")
    data = load_data( # 建立数据 这个是不一样的地方
        data_dir=datapaths,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data, # ATT updated 数据的定义换了 需要额外增加 内容
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        device = device,
        local_rank = local_rank,
        resume_step = get_resume_step(args.resume_checkpoint),
    ).run_loop()

    logger.log("Train Finshed")


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=1000,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        local_rank=0, # ATT updated
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()

import argparse
import os
import tqdm
import logging
import re

from functools import partial
import numpy as np
import torch as th
import torch.distributed as dist
from torchvision import transforms
from torch.utils.data import DataLoader

# from improved_diffusion import dist_util, logger
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.train_util import TrainLoop
from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


# from . import post_processing
from ..utils import *
from ..Data import datasets 
from ..Data.utils import *

logger = logging.getLogger(module_structure(__file__))


class ImprovedDiffusionTransform:

    Parameter_Dict = {
    }

    @classmethod
    def create_INPUT(cls, model_type = "imagenet_64", spacing = "4"):
        model_parameters = cls.Parameter_Dict[model_type]
        MODEL_FLAGS,DIFFUSION_FLAGS,MODEL_PATH,TRAIN_FLAGS = model_parameters["MODEL_FLAGS"],model_parameters["DIFFUSION_FLAGS"],model_parameters["MODEL_PATH"],model_parameters["TRAIN_FLAGS"]
        INPUT = " ".join([MODEL_FLAGS,DIFFUSION_FLAGS,TRAIN_FLAGS,f"--timestep_respacing {spacing}", MODEL_PATH]) 
        INPUT = INPUT.strip()

        INPUT = re.split(r"[ ]+", INPUT)
        return INPUT

    def __init__(self, model_type = "imagenet_64", num_samples = 256, pp = None, mode = "smooth", scale = 8, spacing = "4", ppt = (10000,10000), inner_batch_size = 16, **kwargs):
        '''
            Here size  =  num_samples = batch_size, later to enhance the parallel sampling, this will be changed
        '''

        INPUT = self.create_INPUT(model_type = model_type, spacing = spacing)
        args = self.create_argparser(num_samples, inner_batch_size).parse_args(INPUT)
        args.use_ddim = kwargs["use_ddim"]
        args.device = kwargs["device"]
        args.model_type = model_type

        args.pp = pp
        args.ppt = ppt
        args.mode = mode
        args.scale = scale
        args.alpha = kwargs["alpha"]
        args.frange = kwargs["frange"]
        args.hardness = kwargs["hardness"]

        # dist_util.setup_dist()

        logger.info("creating model and diffusion...")
        model, diffusion = create_model_and_diffusion(
            **args_to_dict(args, model_and_diffusion_defaults().keys())
        )
        # model.load_state_dict(
        #     dist_util.load_state_dict(args.model_path, map_location="cpu")
        # )
        model.load_state_dict(torch.load(args.model_path, map_location="cpu"))

        model.to(args.device)
        model.eval()
        # self.model, self.diffusion, self.dist_util = model, diffusion, dist_util
        self.model, self.diffusion = model, diffusion
        # self.post_processing = vars(post_processing)[post_processing_meth]

        args.num_classes = model.num_classes

        self.args = args

        print("")


    def change_embedding(self, new_classes):
        new_classes_num = len(new_classes)
        ori_weight = self.model.label_emb.weight.data # shape 1000*768
        weight_shape = ori_weight.shape
        self.model.label_emb = th.nn.Embedding(new_classes_num, weight_shape[1]).to(ori_weight.device)
        for e,new_class in enumerate(new_classes):
            if new_class is not None:
                self.model.label_emb.weight.data[e] = ori_weight[new_class]
        self.args.num_classes = len(new_classes)

    def freeze_model(self):
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.label_emb.parameters():
            param.requires_grad = True

    def finetune(self, **kwargs): 
        self.change_embedding(kwargs["data"]["new_classes"]) 
        # self.freeze_model()

        model, diffusion = self.model, self.diffusion

        schedule_sampler = create_named_schedule_sampler("uniform", diffusion) 

        logger.info("creating data loader...")

        dataset = str_get(kwargs["data"]["name"],datasets,**kwargs["data"])
        dataset,_ = leave_one_out(dataset, [], train_indices = kwargs["data"]["train_envs"])
        dataloader = DataLoader(
            datasets.DictDataset(dataset), shuffle = True,
            batch_size=self.args.tr_batch_size, 
            num_workers=kwargs["process"]["num_workers"])
        
        dataiter = recurrent_iter(dataloader)

        logger.info("training...")
        TrainLoop(
            model=model,
            diffusion=diffusion,
            data=dataiter,
            batch_size=self.args.tr_batch_size, 
            microbatch=-1,
            lr=self.args.tr_lr,
            ema_rate="0.9999",
            log_interval=1000,
            save_interval=10000,
            resume_checkpoint="",
            use_fp16=False,
            fp16_scale_growth=1e-3,
            schedule_sampler=schedule_sampler,
            weight_decay=0.0,
            lr_anneal_steps=0,
        ).run_loop()


    def create_argparser(self, num_samples = 256, batch_size = 16):
        defaults = dict(
            clip_denoised=True,
            num_samples=num_samples,
            batch_size=batch_size,
            use_ddim=False,
            model_path="",
            tr_lr=1e-3,
            tr_batch_size=128,
        )
        defaults.update(model_and_diffusion_defaults())
        parser = argparse.ArgumentParser()
        add_dict_to_argparser(parser, defaults)
        return parser

    def model_info(self):
        logger.info("Model")
        logger.info(vars(self.model).keys().__repr__())
        logger.info("Diffusion Model")
        logger.info(vars(self.diffusion).keys().__repr__())
        
        for key in vars(self.diffusion).keys():
            if "time" in key:
                logger.info(f"{key},{getattr(self.diffusion,key)}")
            if "beta" in key:
                logger.info(f"{key},{len(getattr(self.diffusion,key))}")
            
    def addnoise(self, imgs, time_index):

        noise_level = (th.ones(imgs.shape[0]) * time_index).long()
        noise_level = noise_level.to(self.args.device)
        imgs = imgs.to(self.args.device)
        return self.diffusion.q_sample(imgs,noise_level)

    def augment(self, noise, t_start, clear = None, label = None): 

        model, diffusion, args = self.model, self.diffusion, self.args

        logger.info("sampling...")
        all_images = []

        if args.num_classes is not None and label is None:
            gen_label = []

        while len(all_images) * args.batch_size < args.num_samples:
            model_kwargs = {}

            sample_fn = (
                diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
            )
            
            if noise is None:
                noise_batch = None
                batch_size = min(args.batch_size,args.num_samples - len(all_images) * args.batch_size)
            else:
                noise_batch = noise[int(len(all_images) * args.batch_size):int(len(all_images) * args.batch_size+args.batch_size)]
                noise_batch = noise_batch.to(args.device)
                batch_size = noise_batch.shape[0]

            if batch_size == 0:
                break

            if clear is None:
                clear_batch = None
            else:
                clear_batch = clear[int(len(all_images) * args.batch_size):int(len(all_images) * args.batch_size+args.batch_size)]

            if args.num_classes is not None:
                if label is None:
                    classes = th.randint(
                        low=0, high=args.num_classes, size=(batch_size,), device=args.device
                    )
                    gen_label.append(classes.to("cpu"))
                else:
                    classes = label[int(len(all_images) * args.batch_size):int(len(all_images) * args.batch_size+args.batch_size)].to(args.device)
                model_kwargs["y"] = classes

            sample = sample_fn(
                model,
                (batch_size , 3, args.image_size, args.image_size),
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                noise = noise_batch,
                start_t = t_start, 
                pp = args.pp, 
                clear = clear_batch, 
                mode = args.mode, 
                scale = args.scale,
                ppt = args.ppt,
                alpha = args.alpha,
                frange = args.frange,
                hardness = args.hardness,
            )

            all_images.append(sample)
            logger.info(f"created {len(all_images)*args.batch_size} samples")

        arr = th.cat(all_images, dim=0)
        # if args.num_classes is None:
        #     arr_label = None
        # el
        if label is None:
            arr_label = th.cat(gen_label, dim=0)
        else:
            arr_label = label.to("cpu")
        
        return arr, arr_label

    def sample(self):
        return self.augment(noise = None, t_start = None, clear = None, label = None)
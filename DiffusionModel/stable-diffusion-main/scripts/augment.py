"""make variations of input image"""

import argparse, os, sys, glob, random
import PIL
import torch
from torch.utils.data import DataLoader
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torch import autocast
from contextlib import nullcontext
import time
from pytorch_lightning import seed_everything

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.data.datasets import *
from ldm.models.diffusion.pp import *


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

def load_img(path, image_size = 256):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h}) from {path}")
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    # image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = image.resize((image_size, image_size), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1. # 最后是 0 c w h, 区间是 -1 1


def modebasepath(ls,basepath = "~"):
    newls = []
    basepath = basepath.strip("/").split("/")
    for p,l in ls:
        p = p.strip("/").split("/")
        p[:len(basepath)] = basepath
        p = "/" + os.path.join(*p)
        newls.append((p,l))
    return newls

def use_shuffle(dataset, dataset_name, domain_index):
    basepath = r"~\Code\DSI\Data\shuffled_samples"
    filename = f"{dataset_name}_{domain_index}.pt"
    fullpath = os.path.join(basepath, filename)
    if os.path.exists(fullpath):
        samples = torch.load(fullpath,map_location="cpu")
        output = f"shuffled index exists at {fullpath} and loaded"
        print("loaded samples",samples[:5])
        # samples = modebasepath(samples)
        print("base modified samples",samples[:5])
    else:
        samples = dataset.dataset.imgs
        random.shuffle(samples)
        torch.save(samples, fullpath)
        output = f"shuffled index does not exist, is created and saved at {fullpath}"
    dataset.dataset.samples = samples
    dataset.dataset.imgs = samples
    return dataset, output

def step(model, sampler, device, opt, init_image, prompts, st, ed, usenoise):
    init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space
    # print("latent code shape", init_latent.shape)

    sampler.make_schedule(ddim_num_steps=opt.ddim_steps, ddim_eta=opt.ddim_eta, verbose=False)

    precision_scope = autocast if opt.precision == "autocast" else nullcontext

    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                
                # all_samples = list()
                # all_latent = list()
                uc = None
                # if opt.scale != 1.0: # opt.scale 
                #     uc = model.get_learned_conditioning(batch_size * [""])
                if isinstance(prompts, tuple):
                    prompts = list(prompts)
                c = model.get_learned_conditioning(prompts)

                # encode (scaled latent)
                if usenoise:
                    z_enc = sampler.stochastic_encode(init_latent, torch.tensor([st]).to(device))
                else:
                    z_enc = init_latent.to(device)
                
                # decode it

                samples = sampler.decode(z_enc, c, st, unconditional_guidance_scale=opt.scale,
                                        unconditional_conditioning=uc,t_end = ed)

                # print("latent decode shape", samples.shape)

                x_samples = model.decode_first_stage(samples)

                return x_samples

def main():
    parser = argparse.ArgumentParser()

    # parser.add_argument(
    #     "--prompt",
    #     type=str,
    #     nargs="?",
    #     default="a painting of a virus monster playing guitar",
    #     help="the prompt to render"
    # )

    # parser.add_argument(
    #     "--init-img",
    #     type=str,
    #     nargs="?",
    #     help="path to the input image"
    # )

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/img2img-samples"
    )

    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )

    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save indiviual samples. For speed measurements.",
    )

    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )

    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    # parser.add_argument(
    #     "--fixed_code",
    #     action='store_true',
    #     help="if enabled, uses the same starting code across all samples ",
    # )

    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    # parser.add_argument(
    #     "--n_iter",
    #     type=int,
    #     default=1,
    #     help="sample this often",
    # )

    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor, most often 8 or 16",
    )

    # parser.add_argument(
    #     "--n_samples",
    #     type=int,
    #     default=2,
    #     help="how many samples to produce for each given prompt. A.k.a batch size",
    # )

    # parser.add_argument(
    #     "--n_rows",
    #     type=int,
    #     default=0,
    #     help="rows in the grid (default: n_samples)",
    # )

    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )

    parser.add_argument(
        "--strength",
        type=float,
        default=0.75,
        help="strength for noising/unnoising. 1.0 corresponds to full destruction of information in init image",
    )

    # parser.add_argument(
    #     "--from-file",
    #     type=str,
    #     help="if specified, load prompts from this file",
    # )
    parser.add_argument(
        "--config",
        type=str,
        default="logs/f8-kl-clip-encoder-256x256-run1/configs/2022-06-01T22-11-40-project.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="logs/f8-kl-clip-encoder-256x256-run1/checkpoints/last.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--domain_index",
        type=int,
        nargs = "+",
        help="the index of the selected domain",
        default=[0],
    )
    parser.add_argument(
        "--sample_amount",
        type=int,
        help="the number of needed samples",
        default=1024,
    )
    parser.add_argument(
        "--start_time",
        type=int,
        help="the start time",
        default=1,
    )
    parser.add_argument(
        "--freq_range",
        type=float,
        nargs = 2,
        help="diffusion model only gen this part",
        default=[0,1],
    )
    parser.add_argument(
        "--pp_time_range",
        type=int,
        nargs = 2,
        help="pp only in this part",
        default=[0,0],
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="dataset",
        default="PACS",
    )

    opt = parser.parse_args()
    seed_everything(opt.seed)
    opt.strength = opt.start_time / opt.ddim_steps

    assert opt.pp_time_range[1] <= opt.start_time
    assert opt.pp_time_range[0] <= opt.pp_time_range[1]
    assert opt.pp_time_range[0] >= 0
    # opt.pp_time_range = (opt.pp_time_range[0]/opt.ddim_steps, opt.pp_time_range[1]/opt.ddim_steps)

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.plms:
        raise NotImplementedError("check for plms")
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    # batch_size = opt.n_samples
    # n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    # if not opt.from_file:
    #     prompt = opt.prompt
    #     assert prompt is not None
    #     data = [batch_size * [prompt]]

    # else:
    #     print(f"reading prompts from {opt.from_file}")
    #     with open(opt.from_file, "r") as f:
    #         data = f.read().splitlines()
    #         data = list(chunk(data, batch_size))

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path)) # 有多少 sample 了
    grid_count = len(os.listdir(outpath)) - 1 # 有多少 grid了

    if opt.dataset == "PACS_augmentation":
        dataset = onedomainPACS(root = "~/Data/", test_envs = opt.domain_index, data_augmentation = False, image_size = 256)
        dataset, outstr = use_shuffle(dataset, "PACS", opt.domain_index[0])
    print(outstr)
    dataloader = DataLoader(
        dataset, shuffle = False,
        batch_size=64, 
        num_workers=2)

    # assert 0. <= opt.strength <= 1., 'can only work with strength in [0.0, 1.0]'
    # t_enc = int(opt.strength * opt.ddim_steps) 
    t_enc = opt.start_time
    # print(f"target t_enc is {t_enc} steps")

    # pp_start_time_enc = int(opt.pp_time_range[1] * opt.ddim_steps)
    # pp_end_time_enc = int(opt.pp_time_range[0] * opt.ddim_steps)
    pp_start_time_enc = opt.pp_time_range[1] 
    pp_end_time_enc = opt.pp_time_range[0] 

    n_samples = 0
    for e, xy_dict in enumerate(dataloader):

        x = xy_dict["image"]
        prompts = xy_dict["txt"]
        y = xy_dict["y"]
        init_image = x.permute(0,3,1,2).to(device)

        tic = time.time()
        usenoise = True

        if pp_start_time_enc < t_enc:
            x_samples = step(model, sampler, device, opt, init_image, prompts, t_enc, pp_start_time_enc, usenoise)
            usenoise = False
            x_samples = x_samples.float()  
        elif pp_start_time_enc == t_enc:
            x_samples = init_image

        # second stage 

        if pp_end_time_enc < pp_start_time_enc:
            for ppt in range(pp_start_time_enc, pp_end_time_enc, -1):
                x_samples = step(model, sampler, device, opt, x_samples, prompts, ppt, ppt-1, usenoise)
                usenoise = False
                x_samples = pp_frequencydomaincombination(sample = x_samples, 
                                                    origin = init_image, 
                                                    range = opt.freq_range,
                                                    transform = Fourier, inverse_transform = iFourier)
                x_samples = x_samples.float()                                    
        elif pp_end_time_enc == pp_start_time_enc:
            x_samples = x_samples

        # third stage 
        if pp_end_time_enc > 0:
            x_samples = step(model, sampler, device, opt, x_samples, prompts, pp_end_time_enc, 0, usenoise)
            usenoise = False
        elif pp_end_time_enc == 0:
            x_samples = x_samples

        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0) 


        if not opt.skip_save:
            torch.save({"augmented":x_samples.detach().cpu(),"gen_labels":y,},os.path.join(sample_path, f"{e}.pt"))

        clear_aug_samples = torch.zeros(x_samples.shape[0] * 2, * x_samples.shape[1:])

        clear_aug_samples[1::2] = x_samples.detach().cpu()
        clear = torch.clamp((init_image + 1.0) / 2.0, min=0.0, max=1.0)
        clear_aug_samples[0::2] =clear.detach().cpu()
        # all_samples.append(clear_aug_samples)

        if not opt.skip_grid and e < 2:
            # additionally, save as grid
            # grid = torch.stack(all_samples, 0)
            # grid = rearrange(grid, 'n b c h w -> (n b) c h w')
            grid = clear_aug_samples
            grid = make_grid(grid, nrow=2)

            # to image
            grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
            Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
            grid_count += 1
            
        toc = time.time()
        n_samples = n_samples + x_samples.shape[0]
        print(f"iteration {e}, generate {x_samples.shape[0]} in {n_samples}, using {toc - tic} secs.")
        if n_samples > opt.sample_amount:
            break



if __name__ == "__main__":
    main()

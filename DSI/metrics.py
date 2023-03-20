import torch, numpy, torchvision, functools, logging

from torch.utils.data import Dataset as Dataset
# Code for ssim msssim:  https://github.com/VainF/pytorch-msssim
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
# Code for fid kid isc ppl: https://github.com/toshas/torch-fidelity/tree/fb3361aa9d1fcd1e48abb63aab17aef15bc72e28
import torch_fidelity

from .utils import *

logger = logging.getLogger(module_structure(__file__))

def tryexcept(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"{func.__name__} Error: {e}")
            return 0.6180339887
    return wrapper

def detach_var(func):
    @functools.wraps(func)
    def wrapper(*args,**kwargs):
        nargs = [arg.detach() for arg in args]
        nkwargs = {}
        for k,v in kwargs:
            nkwargs[k] = v.detach()
        return func(*nargs, **nkwargs)
    return wrapper

def cpu_var(func):
    @functools.wraps(func)
    def wrapper(*args,**kwargs):
        nargs = [arg.cpu() for arg in args]
        nkwargs = {}
        for k,v in kwargs:
            nkwargs[k] = v.cpu()
        return func(*nargs, **nkwargs)
    return wrapper

def toimage_var(func):
    @functools.wraps(func)
    def wrapper(*args,**kwargs):
        nargs = [(arg*255).to(torch.uint8) for arg in args]
        nkwargs = {}
        for k,v in kwargs:
            nkwargs[k] = (v*255).to(torch.uint8)
        return func(*nargs, **nkwargs)
    return wrapper

class TDataset(Dataset):
    def __init__(self, x):
        super().__init__()
        self.data = x
    def __len__(self):
        return self.data.size(0)
    def __getitem__(self, i ):
        return self.data[i]

@detach_var
@tryexcept
def Correct_Prediction(logits, y):
    preds = torch.argmax(logits, dim = 1)
    return (preds == y).sum().item()
    
@detach_var
@tryexcept
def Correct_Prediction_list(logits, y):
    preds = torch.argmax(logits, dim = 1)
    return (preds == y)

@detach_var
@tryexcept
def Accuracy(logits, y):
    preds = torch.argmax(logits, dim = 1)
    return (preds == y).float().mean().item()

@detach_var
@tryexcept
def psnr(x, y):
    mse = ((x-y)**2).mean()
    upperbound = 1
    return 10*torch.log10(upperbound/mse).cpu().item()

@detach_var
@tryexcept
def msssim(x,y):
    x,y = x*255, y*255
    ms_ssim_val = ms_ssim( x, y, data_range=255, size_average=False )
    return ms_ssim_val.mean().cpu().item()

@detach_var
@tryexcept
def ms_ssim(x,y):
    x,y = x*255, y*255
    ssim_val = ssim( x, y, data_range=255, size_average=False )
    return ssim_val.mean().cpu().item()

@detach_var
@cpu_var
@toimage_var
@tryexcept
def isc(x,y):
    x,y = TDataset(x), TDataset(y)
    metrics_dict = torch_fidelity.calculate_metrics(
        input1=x, 
        input2=y, 
        cuda=True, 
        isc=True, 
        fid=False, 
        kid=False, 
        verbose=False,
        kid_subset_size = len(x)
    )
    return metrics_dict["inception_score_mean"]

@detach_var
@cpu_var
@toimage_var
@tryexcept
def fid(x,y):
    x,y = TDataset(x), TDataset(y)
    metrics_dict = torch_fidelity.calculate_metrics(
        input1=x, 
        input2=y, 
        cuda=True, 
        isc=False, 
        fid=True, 
        kid=False, 
        verbose=False,
        kid_subset_size = len(x)
    )
    return metrics_dict["frechet_inception_distance"]

@detach_var
@cpu_var
@toimage_var
@tryexcept
def kid(x,y):
    x,y = TDataset(x), TDataset(y)
    metrics_dict = torch_fidelity.calculate_metrics(
        input1=x, 
        input2=y, 
        cuda=True, 
        isc=False, 
        fid=False, 
        kid=True, 
        verbose=False, 
        kid_subset_size = len(x)
    )
    return metrics_dict["kernel_inception_distance_mean"]

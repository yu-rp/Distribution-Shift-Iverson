import torch, os, numpy, re
from hashlib import md5
import torch.cuda as cuda
import torch.distributed as distributed

def ptime(second = True):
    if second:
        return datetime.datetime.now().strftime("%d %b. %H:%M:%S")
    else:
        return datetime.datetime.now().strftime("%d %b. %H:%M")
        
def pprint(obj, with_title = False):
    if isinstance(obj,dict):
        if with_title:
            titlelist = [k.__repr__()[:20] for k in obj.keys()]
            valuelist = [v.__repr__()[:20] for v in obj.values()]
            maxlen = max([len(s) for s in titlelist + valuelist]) + 1
            s = ["".join([k.rjust(maxlen) for k in titlelist])]
            s.append("".join([v.rjust(maxlen) for v in valuelist]))
            s = "\n".join(s)
        else:
            valuelist = [v.__repr__()[:20] for v in obj.values()]
            maxlen = max([len(s) for s in valuelist]) + 1
            s = "".join([v.rjust(maxlen) for v in valuelist])
    elif isinstance(obj,(list,tuple)):
        if with_title:
            titlelist = [k.__repr__()[:20] for k in obj[::2]]
            valuelist = [v.__repr__()[:20] for v in obj[1::2]]
            maxlen = max([len(s) for s in titlelist + valuelist]) + 1
            s = ["".join([k.rjust(maxlen) for k in titlelist])]
            s.append("".join([v.rjust(maxlen) for v in valuelist]))
            s = "\n".join(s)
        else:
            valuelist = [v.__repr__()[:20] for v in obj]
            maxlen = max([len(s) for s in valuelist]) + 1
            s = "".join([v.rjust(maxlen) for v in valuelist])
    return s 

def gpu_summary(device):
    return cuda.memory_summary(device)

def str_get(string,module,**kwargs):
    if string in vars(module):
        return vars(module)[string](**kwargs)
    else:
        raise NotImplementedError

def module_structure(file):
    return os.path.relpath(file).split(".")[0].replace("/",".")

def sort_dict(obj):
    if isinstance(obj, list):
        return tuple(sorted(obj))
    elif isinstance(obj, dict):
        tolist = []
        for k in sorted(obj.keys()):
            tolist.append((k,sort_dict(obj[k])))
        return tuple(tolist)
    else:
        return obj

def string_hash(obj):
    return md5(obj.__repr__().encode()).hexdigest()

def establish_communication(**kwargs):
    # # For spawn
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12355'
    # distributed.init_process_group("nccl", rank=rank, world_size=args.num_gpus)
    cuda.set_device(kwargs["device"])
    distributed.init_process_group("nccl",init_method='env://')

def recurrent_iter(obj):
    while True:
        for item in obj:
            yield item

def myround(tens, dec = 3):
    tens = tens * (10 ** dec)
    tens = tens.round()
    tens = tens / (10 ** dec)
    return tens

def printtensor(tensor):
    out = " "
    for t in tensor.tolist():
        out  = out + f",{t:.3f} "
    return out

def image_stat(img):
    img = img.detach().cpu()
    return f"{img.shape},mean {printtensor(img.mean(dim = (0,2,3)))}" \
        f"std {printtensor(img.std(dim = (0,2,3)))}" \
            f"max {printtensor(img.amax(dim = (0,2,3)))}" \
                f"min {printtensor(img.amin(dim = (0,2,3)))}"
import torch

def decay_lr(optimizer, decay_rate):
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay_rate * param_group['lr']
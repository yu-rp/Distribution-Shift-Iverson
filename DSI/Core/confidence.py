import torch

def maximalclassprobability(logit, threshold):
    p = torch.nn.functional.softmax(logit, dim = -1)
    maxp,maxpindex = p.max(dim = 1)
    return maxp > threshold
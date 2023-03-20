import torch

class LayerNorm(torch.nn.Module):
    def __init__(self,dim):
        self.dim = dim
        self.scale = torch.nn.Parameter(torch.randn(*dim))
        self.shift = torch.nn.Parameter(torch.randn(*dim))

    def forward(self,x):
        return x * self.scale + self.shift
    
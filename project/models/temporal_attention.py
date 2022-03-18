import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from spikingjelly.clock_driven import surrogate, layer, neuron, functional

from project.models.liaf import MultiStepLIAFNode


class SqueezeAndExcite(nn.Module):
    """Some Information about SqueezeAndExcite"""

    def __init__(self, T: int, r: int = 1, d_th: float = 1.0):
        super(SqueezeAndExcite, self).__init__()
        self.r = r
        self.d_th = d_th

        # excite weights
        self.w1 = nn.Linear(T, T // r, bias=False)
        self.w2 = nn.Linear(T // r, T, bias=False)

    def forward(self, x):
        s = self.squeeze(x)
        d = self.excite(s)
        
        # put d (excite) score tensor to same size as input x
        d = d.view(d.shape[0], d.shape[1], 1, 1, 1).expand_as(x) # d of shape = (T,B,C,H,W) here
        
        return d * x

    def squeeze(self, x):
        s = torch.zeros((x.shape[0], x.shape[1]))  # shape (T, B)

        for t in range(x.shape[0]):
            s[t] = x[t].mean(0)

        return s

    def excite(self, s: torch.Tensor):
        d = s.permute((1, 0))  # shape = (B, T)

        # apply excite w1 + relu
        d = torch.relu(self.w1(d))

        # apply excite w2 + sigmoid
        d = torch.sigmoid(self.w2(d))

        return d.permute(1, 0)  # shape = (T, B)
    
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from spikingjelly.clock_driven import surrogate, neuron, functional, layer

class SigmaDeltaNeuron(nn.Module):
    """Some Information about SigmaDeltaNeuron"""
    def __init__(self, threshold: float = 1., ):
        super(SigmaDeltaNeuron, self).__init__()
        self.memory = {}
        
        # register internal state for sigma value

    def forward(self, x):
        # SIGMA
        ## Sum input with internal state
        
        
        

        return x
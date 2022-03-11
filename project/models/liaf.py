from typing import Callable
from spikingjelly.clock_driven import surrogate, neuron, functional, layer
import torch


class LIAFNode(neuron.BaseNode):
    def __init__(self, tau: float = 2., v_threshold: float = 1.,
                 v_reset: float = 0., surrogate_function: Callable = surrogate.Sigmoid(),
                 detach_reset: bool = False):
        assert isinstance(tau, float) and tau > 1.

        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset)
        self.tau = tau

    def extra_repr(self):
        return super().extra_repr() + f', tau={self.tau}'

    def neuronal_charge(self, x: torch.Tensor):
        if self.v_reset is None:
            self.v = self.v + (x - self.v) / self.tau

        else:
            if isinstance(self.v_reset, float) and self.v_reset == 0.:
                self.v = self.v + (x - self.v) / self.tau
            else:
                self.v = self.v + (x - (self.v - self.v_reset)) / self.tau

    def forward(self, x: torch.Tensor):
        self.neuronal_charge(x)
        analog_charge = torch.relu(self.v)
        self.neuronal_fire()
        self.neuronal_reset()
        return analog_charge


class MultiStepLIAFNode(LIAFNode):
    def __init__(self, tau: float = 2., v_threshold: float = 1.,
                 v_reset: float = 0., surrogate_function: Callable = surrogate.Sigmoid(),
                 detach_reset: bool = False):
        super().__init__(tau, v_threshold, v_reset, surrogate_function, detach_reset)
        self.register_memory('v_seq', None)
        self.register_memory('spike_seq', None)

    def forward(self, x_seq: torch.Tensor):
        assert x_seq.dim() > 1
        # x_seq.shape = [T, *]
        self.v_seq = torch.zeros_like(x_seq.data)
        self.spike_seq = torch.zeros_like(x_seq.data)

        for t in range(x_seq.shape[0]):
            self.spike_seq[t] = super().forward(x_seq[t])
            self.v_seq[t] = self.v
        return self.spike_seq

    def extra_repr(self):
        return super().extra_repr() + f', backend={self.backend}'

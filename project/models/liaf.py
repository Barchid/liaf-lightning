from typing import Callable
from spikingjelly.clock_driven import surrogate, neuron, functional, layer
import torch
try:
    import cupy
    from . import neuron_kernel, cu_kernel_opt
except ImportError:
    neuron_kernel = None


class LIAFNode(neuron.BaseNode):
    def __init__(self, tau: float = 2., v_threshold: float = 1.,
                 v_reset: float = 0., surrogate_function: Callable = surrogate.Sigmoid(),
                 detach_reset: bool = False):
        """
        * :ref:`API in English <LIFNode.__init__-en>`

        .. _LIFNode.__init__-cn:

        :param tau: 膜电位时间常数
        :type tau: float

        :param v_threshold: 神经元的阈值电压
        :type v_threshold: float

        :param v_reset: 神经元的重置电压。如果不为 ``None``，当神经元释放脉冲后，电压会被重置为 ``v_reset``；
            如果设置为 ``None``，则电压会被减去 ``v_threshold``
        :type v_reset: float

        :param surrogate_function: 反向传播时用来计算脉冲函数梯度的替代函数
        :type surrogate_function: Callable

        :param detach_reset: 是否将reset过程的计算图分离
        :type detach_reset: bool


        Leaky Integrate-and-Fire 神经元模型，可以看作是带漏电的积分器。其阈下神经动力学方程为：

        .. math::
            V[t] = V[t-1] + \\frac{1}{\\tau}(X[t] - (V[t-1] - V_{reset})

        * :ref:`中文API <LIFNode.__init__-cn>`

        .. _LIFNode.__init__-en:

        :param tau: membrane time constant
        :type tau: float

        :param v_threshold: threshold voltage of neurons
        :type v_threshold: float

        :param v_reset: reset voltage of neurons. If not ``None``, voltage of neurons that just fired spikes will be set to
            ``v_reset``. If ``None``, voltage of neurons that just fired spikes will subtract ``v_threshold``
        :type v_reset: float

        :param surrogate_function: surrogate function for replacing gradient of spiking functions during back-propagation
        :type surrogate_function: Callable

        :param detach_reset: whether detach the computation graph of reset
        :type detach_reset: bool

        The Leaky Integrate-and-Fire neuron, which can be seen as a leaky integrator.
        The subthreshold neural dynamics of it is as followed:

        .. math::
            V[t] = V[t-1] + \\frac{1}{\\tau}(X[t] - (V[t-1] - V_{reset})
        """
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
        """

        * :ref:`API in English <BaseNode.forward-en>`

        .. _BaseNode.forward-cn:

        :param x: 输入到神经元的电压增量
        :type x: torch.Tensor

        :return: 神经元的输出脉冲
        :rtype: torch.Tensor

        按照充电、放电、重置的顺序进行前向传播。

        * :ref:`中文API <BaseNode.forward-cn>`

        .. _BaseNode.forward-en:

        :param x: increment of voltage inputted to neurons
        :type x: torch.Tensor

        :return: out spikes of neurons
        :rtype: torch.Tensor

        Forward by the order of `neuronal_charge`, `neuronal_fire`, and `neuronal_reset`.

        """
        self.neuronal_charge(x)
        analog_charge = torch.relu(self.v)
        self.neuronal_fire()
        self.neuronal_reset()
        return analog_charge


class MultiStepLIAFNode(LIAFNode):
    def __init__(self, tau: float = 2., v_threshold: float = 1.,
                 v_reset: float = 0., surrogate_function: Callable = surrogate.Sigmoid(),
                 detach_reset: bool = False, backend='torch'):
        """
        * :ref:`API in English <MultiStepLIFNode.__init__-en>`

        .. _MultiStepLIAFNode.__init__-cn:

        :param tau: 膜电位时间常数
        :type tau: float

        :param v_threshold: 神经元的阈值电压
        :type v_threshold: float

        :param v_reset: 神经元的重置电压。如果不为 ``None``，当神经元释放脉冲后，电压会被重置为 ``v_reset``；
            如果设置为 ``None``，则电压会被减去 ``v_threshold``
        :type v_reset: float

        :param surrogate_function: 反向传播时用来计算脉冲函数梯度的替代函数
        :type surrogate_function: Callable

        :param detach_reset: 是否将reset过程的计算图分离
        :type detach_reset: bool

        :param backend: 使用哪种计算后端，可以为 ``'torch'`` 或 ``'cupy'``。``'cupy'`` 速度更快，但仅支持GPU。
        :type backend: str

        多步版本的 :class:`spikingjelly.clock_driven.neuron.LIFNode`。

        .. tip::

            对于多步神经元，输入 ``x_seq.shape = [T, *]``，不仅可以使用 ``.v`` 和 ``.spike`` 获取 ``t = T - 1`` 时刻的电压和脉冲，还能够
            使用 ``.v_seq`` 和 ``.spike_seq`` 获取完整的 ``T`` 个时刻的电压和脉冲。

        .. tip::

            阅读 :doc:`传播模式 <./clock_driven/10_propagation_pattern>` 以获取更多关于单步和多步传播的信息。

        * :ref:`中文API <MultiStepLIFNode.__init__-cn>`

        .. _MultiStepLIFNode.__init__-en:

        :param tau: membrane time constant
        :type tau: float

        :param v_threshold: threshold voltage of neurons
        :type v_threshold: float

        :param v_reset: reset voltage of neurons. If not ``None``, voltage of neurons that just fired spikes will be set to
            ``v_reset``. If ``None``, voltage of neurons that just fired spikes will subtract ``v_threshold``
        :type v_reset: float

        :param surrogate_function: surrogate function for replacing gradient of spiking functions during back-propagation
        :type surrogate_function: Callable

        :param detach_reset: whether detach the computation graph of reset
        :type detach_reset: bool

        :param backend: use which backend, ``'torch'`` or ``'cupy'``. ``'cupy'`` is faster but only supports GPU
        :type backend: str

        The multi-step version of :class:`spikingjelly.clock_driven.neuron.LIFNode`.

        .. admonition:: Tip
            :class: tip

            The input for multi-step neurons are ``x_seq.shape = [T, *]``. We can get membrane potential and spike at
            time-step ``t = T - 1`` by ``.v`` and ``.spike``. We can also get membrane potential and spike at all ``T``
            time-steps by ``.v_seq`` and ``.spike_seq``.

        .. admonition:: Tip
            :class: tip

            Read :doc:`Propagation Pattern <./clock_driven_en/10_propagation_pattern>` for more details about single-step
            and multi-step propagation.

        """
        super().__init__(tau, v_threshold, v_reset, surrogate_function, detach_reset)
        self.register_memory('v_seq', None)
        self.register_memory('spike_seq', None)

        assert backend == 'torch' or backend == 'cupy'
        assert not (backend == 'cupy' and neuron_kernel is None), 'cupy is not installed'
        self.backend = backend

    def forward(self, x_seq: torch.Tensor):
        assert x_seq.dim() > 1
        # x_seq.shape = [T, *]
        self.v_seq = torch.zeros_like(x_seq.data)
        self.spike_seq = torch.zeros_like(x_seq.data)

        if self.backend == 'torch':
            for t in range(x_seq.shape[0]):
                self.spike_seq[t] = super().forward(x_seq[t])
                self.v_seq[t] = self.v
            return self.spike_seq

        elif self.backend == 'cupy':
            if isinstance(self.v, float):
                v_init = self.v
                self.v = torch.zeros_like(x_seq[0].data)
                if v_init != 0.:
                    torch.fill_(self.v, v_init)

            self.spike_seq, self.v_seq = neuron_kernel.MultiStepLIFNodePTT.apply(
                x_seq.flatten(1), self.v.flatten(0), self.tau, self.v_threshold, self.v_reset, self.detach_reset, self.surrogate_function.cuda_code)

            self.spike_seq = self.spike_seq.reshape(x_seq.shape)
            self.v_seq = self.v_seq.reshape(x_seq.shape)

            self.spike = self.spike_seq[-1].clone()
            self.v = self.v_seq[-1].clone()

            return self.spike_seq
        else:
            raise NotImplementedError

    def extra_repr(self):
        return super().extra_repr() + f', backend={self.backend}'

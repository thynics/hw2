"""The module.
"""
import math
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight =Parameter(init.kaiming_uniform(in_features, out_features), dtype=dtype, device=device)
        if bias:
            ub = init.kaiming_uniform(out_features, 1)
            ub = ops.reshape(ub, (1, out_features))
            self.bias = Parameter(ub, dtype=dtype, device=device)

        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        z1 = ops.matmul(X, self.weight)
        if self.bias is not None:
            brod_bias = ops.broadcast_to(self.bias, z1.shape)
            z1 = z1 + brod_bias
        return z1
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        d1 = X.shape[0]
        d2 = int(math.prod(X.shape) / d1)
        return X.reshape((d1, d2))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION

class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for m in self.modules:
            if isinstance(m, Module):
                x = m.forward(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        zy = ops.summation(logits * init.one_hot(logits.shape[-1], y), axes=1).reshape((-1, 1))
        lse = ops.logsumexp(logits, axes=1).reshape((-1, 1))
        return ops.summation(lse - zy) / lse.shape[0]
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.device = device
        self.dtype = dtype
        self.weight = Parameter(init.ones(1, dim, device=device, dtype=dtype), device=device, dtype=dtype)
        self.bias =  Parameter(init.zeros(1, dim, device=device, dtype=dtype), device=device, dtype=dtype)
        self.running_mean = Parameter(init.zeros(dim, device=device, dtype=dtype), device=device, dtype=dtype)
        self.running_var = Parameter(init.ones(dim, device=device, dtype=dtype), device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.weight.shape != (1, self.dim):
            self.weight = self.weight.reshape((1, self.dim))
        if self.bias.shape != (1, self.dim):
            self.bias = self.bias.reshape((1, self.dim))
        if self.training:
            batch_size, feature_size = x.shape
            mean = (x.sum(axes=(0, )) / batch_size).reshape((1, feature_size))
            var = (((x - mean.broadcast_to(x.shape)) ** 2).sum(axes=(0, )) / batch_size).reshape((1, feature_size))
            self.running_mean = self.running_mean *(1 - self.momentum) + mean.reshape(self.running_mean.shape) * ( self.momentum)
            self.running_var = self.running_var *(1 - self.momentum) + var.reshape(self.running_var.shape) * (self.momentum)
            mean = mean.broadcast_to(x.shape)
            var = var.broadcast_to(x.shape)
            std_x = (x - mean) / ops.power_scalar(var + self.eps, 0.5)
            weight = self.weight.broadcast_to(x.shape)
            bias = self.bias.broadcast_to(x.shape)
            return std_x * weight + bias
        else:
            std_x = (x - self.running_mean.broadcast_to(x.shape)) / ops.power_scalar(self.running_var.broadcast_to(x.shape) + self.eps, 0.5)
            return std_x * self.weight.broadcast_to(x.shape) + self.bias.broadcast_to(x.shape)
        ### END YOUR SOLUTION



class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = Parameter(init.ones(1, dim, device=device, dtype=dtype), device=device, dtype=dtype)
        ### BEGIN YOUR SOLUTION
        self.bias = Parameter(init.zeros(1, dim, device=device, dtype=dtype), device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch_size, feature_size = x.shape
        mean = (x.sum(axes=(1, )) / feature_size).reshape((batch_size, 1)).broadcast_to(x.shape)
        var = (((x - mean) ** 2).sum(axes=(1, )) / feature_size).reshape((batch_size, 1)).broadcast_to(x.shape)
        std_x = (x - mean) / ops.power_scalar(var + self.eps, 0.5)
        weight = self.weight.broadcast_to(x.shape)
        bias = self.bias.broadcast_to(x.shape)
        return std_x * weight + bias
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            mask = (np.random.rand(*x.shape) < self.p) / ( 1 - self.p)
            return x * mask
        else:
            return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x) + x
        ### END YOUR SOLUTION

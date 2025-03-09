from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        max_z = array_api.max(Z, keepdims=True, axis=1)
        lse = array_api.log(array_api.sum(array_api.exp(Z - max_z), axis=1)) + max_z.squeeze()
        lse = lse.reshape(-1, 1)
        return Z - lse
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        z = node.inputs[0]
        lse = z - node  # lse = z - (z - lse) → lse正确
        softmax_z = exp(z - lse.broadcast_to(z.shape))  # 计算softmax
        sum_out_grad = summation(out_grad, axes=1).reshape((-1, 1))
        return out_grad - softmax_z * sum_out_grad
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        max_z = array_api.max(Z, axis=self.axes, keepdims=True)
        return array_api.log(array_api.sum(array_api.exp(Z - max_z), axis=self.axes)) + max_z.squeeze()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        if self.axes is None:
            self.axes = tuple(range(len(node.inputs[0].shape)))
        if isinstance(self.axes, int):
            self.axes = (self.axes,)
        z = node.inputs[0]
        shape = [1 if i in self.axes else z.shape[i] for i in range(len(z.shape))]
        gradient = exp(z - node.reshape(shape).broadcast_to(z.shape))

        # exp(z - node.reuslt)

        return out_grad.reshape(shape).broadcast_to(z.shape)*gradient


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)


"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for p in self.params:
            if p.grad is None:
                continue
            grad = p.grad
            if self.weight_decay > 0.0:
                grad += self.weight_decay * p.data
            if self.momentum > 0.0:
                if p not in self.u:
                    self.u[p] = ndl.zeros_like(p.data)
                self.u[p] = self.momentum * self.u[p] + (1 - self.momentum) * grad
                p.data -= self.lr * self.u[p]
            else:
                p.data -= self.lr * grad
        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        ### BEGIN YOUR SOLUTION
        total = 0.
        for p in self.params:
            if p.grad is not None:
                total += p.grad.data.norm()
        if total > max_norm:
            scale = max_norm / total
            for p in self.params:
                if p.grad is not None:
                    p.grad.data.mul_(scale)
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        for param in self.params:
            if param.grad is not None:
                grad = param.grad.data
                if param not in self.m.keys():
                    self.m[param] = ndl.zeros_like(grad)
                if param not in self.v.keys():
                    self.v[param] = ndl.zeros_like(grad)
                grad += self.weight_decay * param
                self.m[param] = ((self.beta1 * self.m[param]) + (1 - self.beta1) * grad.data)
                self.v[param] = ((self.beta2 * self.v[param]) + (1 - self.beta2) * grad * grad)
                u_hat = (self.m[param]/ (1 - self.beta1 ** self.t))
                v_hat = (self.v[param]/ (1 - self.beta2 ** self.t))
                param.data -= (self.lr * u_hat / (ndl.ops.power_scalar(v_hat, 0.5).data + self.eps))

        ### END YOUR SOLUTION

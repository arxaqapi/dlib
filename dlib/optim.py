"""
An optimizer is used to adjust the param
of our network (weights and biases)
based on the gradient computed during backprop
"""
from dlib.nn import NeuralNet


class Optimizer:
    def step(self, net: NeuralNet) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, lr: float = 0.01) -> None:
        self.lr = lr
    
    def step(self, net: NeuralNet) -> None:
        for param, grad in net.params_and_grads():
            param -= self.lr * grad
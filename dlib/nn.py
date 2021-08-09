"""
A neural net is a collection of layers.
It behaves a lot like a layer itself, although
we're not going to make it one.
"""

from typing import Iterator, Sequence, Tuple

from dlib.tensor import Tensor
from dlib.layers import Layer


class NeuralNet:
    def __init__(self, layers: Sequence[Layer]) -> None:
        self.layers = layers
    
    def forward(self, inputs: Tensor) -> Tensor:
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs
    
    def backward(self, grad: Tensor) -> Tensor:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad
    
    def params_and_grads(self) -> Iterator[Tuple[Tensor, Tensor]]:
        for layer in self.layers:
            # print(layer)
            # name = 'w' then 'b' | # param is an array with parameters
            for name, param in layer.params.items():
                grad = layer.grads[name]
                yield param, grad
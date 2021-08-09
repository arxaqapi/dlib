"""
A neural network is made out of layers.
Each layer needs to pass its inputs forward
and propagate gradients backward. For example,
a neural net might look like:

input -> linear -> tanh -> linear -> output
"""
from typing import Callable, Dict
import numpy as np

from dlib.tensor import Tensor


class Layer:
    def __init__(self) -> None:
        self.params: Dict[str, Tensor] = {}
        self.grads: Dict[str, Tensor] = {}
        self.inputs: Tensor = None

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Produce the outputs corresponding to these inputs
        """
        raise NotImplementedError

    def backward(self, grad: Tensor) -> Tensor:
        """
        Backpropagate this gradient through the layer
        (gradient is the partial derivatives of some function (the loss functions))
        """
        raise NotImplementedError
    
    def __str__(self) -> str:
        return f"[-{self.inputs=};\n{self.grads=};\n{self.params=}-]"


class Linear(Layer):
    """
    computes output = inputs @ w + b
    """
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        # inputs will be (batch_size, input_size)
        # outputs will be (batch_size, output_size)
        self.params['w'] = np.random.randn(input_size, output_size)
        self.params['b'] = np.random.randn(output_size)
    
    def forward(self, inputs: Tensor) -> Tensor:
        """
        outputs = inputs @ w + b
        """
        # assert inputs.shape[1] == self.input_size
        self.inputs = inputs # save a copy of the inputs for backpropagation
        return inputs @ self.params['w'] + self.params['b']
    
    def backward(self, grad: Tensor) -> Tensor:
        """
        if y = f(x) and x = a * b + c
        then dy/da = f'(x) * b
        and dy/db = f'(x) * a
        and dy/dc = f'(x)

        if y = f(x) and x = a @ b + c
        then dy/da = f'(x) @ b.T
        then dy/db = a.T @ f'(x)
        then dy/dc = f'(x) 
        """
        self.grads['b'] = np.sum(grad, axis=0)
        self.grads['w'] = self.inputs.T @ grad
        return grad @ self.params['w'].T


F = Callable[[Tensor], Tensor]

class Activation(Layer):
    """
    An activation layer applies a fonction
    elementwise to its inputs
    """
    def __init__(self, f: F, f_prime: F) -> None:
        super().__init__()
        self.f = f
        self.f_prime = f_prime
    
    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        return self.f(inputs)
    
    def backward(self, grad: Tensor) -> Tensor:
        """
        if y = f(x) and x = g(z) | y = f(g(z))
        then dy/dz = f'(x) * g'(z)
        """
        return self.f_prime(self.inputs) * grad


def tanh(x: Tensor) -> Tensor:
    return np.tanh(x)

def tanh_prime(x: Tensor) -> Tensor:
    return 1 - tanh(x) ** 2

class Tanh(Activation):
    def __init__(self) -> None:
        super().__init__(tanh, tanh_prime)


def sigmoid(x: Tensor) -> Tensor:
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x: Tensor) -> Tensor:
    return sigmoid(x) * (1 - sigmoid(x))

class Sigmoid(Activation):
    def __init__(self) -> None:
        super().__init__(sigmoid, sigmoid_prime)


def relu(x: Tensor) -> Tensor:
    return np.max(0, x)

def relu_prime(x: Tensor) -> Tensor:
    return np.array( [1 if e >= 0 else 0 for e in x] )

class ReLU(Activation):
    def __init__(self) -> None:
        super().__init__(relu, relu_prime)
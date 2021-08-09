"""
A loss functions measures how good our predictions are,
the value is then used to adjust the parameters of our network
"""
import numpy as np

from dlib.tensor import Tensor

class Loss:
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        raise NotImplementedError
    
    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        raise NotImplementedError


class MSE(Loss):
    """
    MSE is mean squared error
    """
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        # n = predicted.shape[0] | actual.shape[0]
        return np.sum((predicted - actual) ** 2) / actual.shape[0]
    
    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        return 2 * (predicted - actual)
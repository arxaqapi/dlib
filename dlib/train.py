"""
Function to train a neural net
"""

from typing import List
from dlib.tensor import Tensor
from dlib.nn import NeuralNet
from dlib.loss import Loss, MSE
from dlib.optim import Optimizer, SGD
from dlib.data import DataIterator, BatchIterator

def train(net: NeuralNet,
          inputs: Tensor,
          targets: Tensor,
          num_epochs: int = 5000,
          iterator: DataIterator = BatchIterator(),
          loss: Loss = MSE(),
          optimizer: Optimizer = SGD()) -> List[float]:
    print("------ Start ------\n")
    elosses = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in iterator(inputs, targets):
            predicted = net.forward(batch.inputs)
            epoch_loss += loss.loss(predicted, batch.targets)
            grad = loss.grad(predicted, batch.targets)
            net.backward(grad)
            optimizer.step(net)
        elosses.append(epoch_loss)
        print(f"epoch: {epoch} | epoch_loss: {epoch_loss}")
    print("------ End ------\n")
    return elosses
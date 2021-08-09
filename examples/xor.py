"""
Example of a function that can't be
learned with a simple linear model is XOR
""" 
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from dlib.train import train
from dlib.nn import NeuralNet
from dlib.layers import Linear, Tanh
from dlib.optim import SGD

inputs = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

targets = np.array([
    # is 0, is 1
    [1, 0],
    [0, 1],
    [0, 1],
    [1, 0],
])

net = NeuralNet([
    Linear(input_size=2, output_size=2),
    Tanh(),
    Linear(input_size=2, output_size=2)
])

train(
    net,
    inputs=inputs,
    targets=targets,
    # num_epochs=1000,
    # optimizer=SGD(lr=0.1)
)

for x, y in zip(inputs, targets):
    predicted = net.forward(x)
    print(f"x: {x}, predicted: {predicted}, y: {y}")
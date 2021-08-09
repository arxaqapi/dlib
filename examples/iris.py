import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.preprocessing import normalize 
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from dlib.train import train
from dlib.nn import NeuralNet
from dlib.layers import Linear, Tanh, Sigmoid

# 4 features, 3 classes
inputs, target = load_iris(return_X_y=True)
inputs = normalize(inputs, axis=0)
# one hot encoding
targets = np.array( [np.eye(3)[c] for c in target] )
# shuffle and split dataset
inputs, targets = shuffle(inputs, targets)
X_train, X_test, y_train, y_test = train_test_split(inputs, targets, test_size=0.2)


net = NeuralNet([
    Linear(input_size=4, output_size=20),
    Tanh(),
    Linear(input_size=20, output_size=20),
    Tanh(),
    Linear(input_size=20, output_size=3),
    Sigmoid()
])

eloss = train(
    net,
    inputs=X_train,
    targets=y_train,
    num_epochs=1000
)

plt.plot(eloss, 'r')
plt.show()

y_true, y_pred = [], []

for x, y in zip(X_test, y_test):
    predicted = net.forward(x).round(decimals=3)
    y_pred.append(np.argmax(y))
    y_true.append(np.argmax(predicted))    
    print(f"x: {x}, predicted: {predicted}, y: {y}")

print(classification_report(y_true=y_true, y_pred=y_pred, target_names=['setosa', 'versicolor', 'virginica']))
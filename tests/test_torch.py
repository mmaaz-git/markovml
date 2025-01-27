import os, sys; sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Sequential
from markovml.markovml import MarkovReward
from markovml.utils.models_ext import SequentialClassifier
import gurobipy as gp
from gurobipy import GRB

def train_simple_nn(model, X, y, is_classifier=False, num_classes=None, epochs=10):
    optimizer = torch.optim.Adam(model.parameters())
    if is_classifier:
        if num_classes==2:
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()

    X_tensor = torch.FloatTensor(X)
    if is_classifier:
        y_tensor = torch.LongTensor(y) if num_classes>2 else torch.FloatTensor(y).view(-1, 1)
    else:
        y_tensor = torch.FloatTensor(y)

    for i in range(epochs):
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = criterion(output, y_tensor)
        loss.backward()
        optimizer.step()

        print(f"Epoch {i+1} loss: {loss.item()}")

    model.eval()
    return model

# Create synthetic data for training the transition model
X = np.random.rand(1000, 5)  # 5 features for example
y = np.random.randint(0, 2, size=1000)

# Create and train transition model
trans_model = Sequential(
    nn.Linear(5, 8),
    nn.ReLU(),
    nn.Linear(8, 1)  # 2 outputs for binary classification
)
trans_model = train_simple_nn(trans_model, X, y, is_classifier=True, num_classes=2, epochs=5)
trans_model = SequentialClassifier(trans_model)


# Create MarkovReward object
mrp = MarkovReward(n_states=2, n_features=5)
mrp.add_ml_model(trans_model)
mrp.set_pi([1.0, 0.0])
mrp.set_r([10.0, 0.0])
mrp.set_P([[1 - mrp.ml_outputs[0][0], mrp.ml_outputs[0][0]], [0.0, 1.0]])

for i in range(5):
    mrp.features[i].LB = 0
    mrp.features[i].UB = 100

for i in range(1, 5):
    mrp.features[i].VType = GRB.BINARY

print(mrp.optimize(use_decomp=True, verbose=True, sense="max"))

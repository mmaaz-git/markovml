"""
This module contains several utilities that are necessary and useful for building Markov processes with learned parameters.

In `gurobi_ml_ext`, we extend the Gurobi machine learning functionality to work with additional models, namely classifiers with softmax outputs.

In `models_ext`, we contain some additional "models" that we have implemented for convenience, namely `DecisionRules` which allows the user to specify "if-then" statements in natural language, and `SequentialClassifier`, which is a wrapper for a PyTorch `nn.Sequential` model that adds a softmax layer at the end of it.

In `ima`, we contain all the functions necessary for *interval matrix analysis*, culminating in the implementation of the interval version of the Gauss-Seidel method.

"""
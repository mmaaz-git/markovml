"""
This is the documentation for `markovml`.

The main functionality is contained in the `markovml` module. It contains three classes: `MarkovReward`, `MarkovReach`, and `MarkovHitting`, which let you analyze the total reward, reachability, and hitting time of a Markov process with learned parameters. They all are based on the `AbstractMarkov` class, which provides the common functionality for all three classes.

Currently, the following models are supported:
- `sklearn.linear_model.LinearRegression`
- `sklearn.linear_model.Ridge`
- `sklearn.linear_model.Lasso`
- `sklearn.linear_model.LogisticRegression`
- `sklearn.tree.DecisionTreeRegressor`
- `sklearn.tree.DecisionTreeClassifier`
- `sklearn.ensemble.RandomForestRegressor`
- `sklearn.ensemble.RandomForestClassifier`
- `sklearn.neural_network.MLPRegressor`
- `sklearn.neural_network.MLPClassifier`
- `markovml.utils.models_ext.DecisionRules` -- our custom model for "if-then" rules, e.g., "if age > 65 then 3.0"
- `torch.nn.Sequential` -- only Linear and ReLU layers are supported
- `markovml.utils.models_ext.SequentialClassifier` -- wrapper around a PyTorch Sequential model that adds a final softmax layer

In the `utils` module, you can find some utility functions for working with Markov processes. Namely, there are the modules `utils.gurobi_ml_ext`, `utils.models_ext`, and `utils.ima`. The first one extends the Gurobi machine learning functionality to work with additional models, namely classifiers with softmax outputs; the second one contains some additional "models" that we have implemented for convenience, namely `DecisionRules` which allows the user to specify "if-then" statements in natural language, and `SequentialClassifier`, which is a wrapper for a PyTorch `nn.Sequential` model that adds a softmax layer at the end of it; the third one contains all the functions necessary for *interval matrix analysis*, culminating in the implementation of the interval version of the Gauss-Seidel method.
"""

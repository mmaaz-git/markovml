# markovml

markovml is a Python package for formally verifying properties of Markov processes with learned parameters.
It accompanies the paper "Formal Verification of Markov Processes with Learned Parameters".

More precisely, it allows you to build Markov processes whose parameters are related to an exogenous feature vector through machine learning models, and verify the reachability, hitting time, and total reward of the process. It formulates these problems as bilinear programs, which we solve efficiently using a special decomposition and bound propagation scheme. For a list of supported models, see the documentation.

It is built on top of [Gurobi](https://www.gurobi.com/). Gurobi is typically a commercial solver, but there is also a free version. For small state spaces and small models, it is possible to use the free version, which is installed if you simply `pip install gurobipy`. For academics, you can obtain a free academic license which will give you full access to the solver.

## Installation

The package is in the `markovml/` directory. You can simply add it to your Python path and import it or put it in your project directory. It requires `gurobipy==12.0.0` and `gurobi-machinelearning==1.5.1`. Depending on which ML models you want to use, you also need `scikit-learn` and/or `torch`.

## Documentation

See the documentation in `docs/`. They are html files that you can open in your browser.

## Examples

Here is a minimal example of how to use the package. We build a simple two-state Markov reward process and embed a logistic regression model into it. Then, we maximize the total reward.

```python
from markovml.markovml import MarkovReward
from sklearn.linear_model import LogisticRegression
import numpy as np

# Create a Markov process with 2 states and 2 features
mrp = MarkovReward(n_states=2, n_features=2)

# fix some of the parameters
mrp.set_r([1, 0])
mrp.set_pi([1, 0])

# train a classifier
X = np.random.rand(100, 2)
y = np.random.randint(0, 2, 100)
clf = LogisticRegression().fit(X, y)

# add the classifier to the Markov process
mrp.add_ml_model(clf)

# link it to the transition probabilities
# this means the first output of the first ML model is the probability of transitioning to the second state
mrp.set_P([[1 - mrp.ml_outputs[0][0], mrp.ml_outputs[0][0]], [0, 1]])

# set bounds on the features
mrp.add_feature_constraint(mrp.features[0] >= 65)
mrp.add_feature_constraint(mrp.features[1] >= 100)

# optimize the reward
mrp.optimize(sense="max")
```

For a full walkthrough of the package, see the tutorials in `tutorials/`.

## Paper experiments

In our paper, we perform computational experiments testing our decomposition and bound propagation scheme against directly solving the bilinear programs. To reproduce the results, run the experiments in `experiments/experiments.py`. This takes several possible arguments as described below.

```
usage: experiments.py [-h] [--all] [--states] [--models] [--trees] [--nns] [--ablation] [--small] [--verbose] [--logfile LOGFILE] [--output-dir OUTPUT_DIR]

Run MarkovML experiments

options:
  -h, --help            show this help message and exit
  --all                 Run all experiments
  --states              Run state scaling experiment
  --models              Run model count scaling experiment
  --trees               Run decision tree depth scaling experiment
  --nns                 Run neural network architecture scaling experiment
  --ablation            Run ablation studies
  --small               Run small version of experiments
  --verbose             Enable verbose output
  --logfile LOGFILE     Path to Gurobi log file (default: None)
  --output-dir OUTPUT_DIR
                        Directory to store results (default: results)
```

Running `--all` will run all the experiments shown in the paper. This requires a full Gurobi license, and takes several days to run. However, you can run the `--small` version on a free license. These experiments take only a couple of minutes to run, and already give a strong indication of the performance of our algorithm (at least a 10x speedup over direct solving).

The full results of the experiments from the paper are in the `experiments/paperresults/` directory, which also contains the code used to generate the plots and tables.
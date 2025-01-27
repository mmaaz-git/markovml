import os, sys; sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
import pandas as pd
from time import time
import warnings
from contextlib import redirect_stdout, nullcontext
from markovml.markovml import MarkovReward
import os
import argparse

warnings.filterwarnings("ignore")

class AblatedMarkov(MarkovReward):
    """Wrapper class that allows ablation of optimization steps"""

    def __init__(self, *args, skip_from_step=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.skip_from_step = skip_from_step

    def optimize(self, use_decomp=True, sense="max", verbose=False, gurobi_params=None):
        """
        Override optimize to allow skipping steps from a certain point onwards.

        Args:
            skip_from_step (str): Step from which to start skipping. Options:
                - 'ml_bounds': Skip ML bounds computation and subsequent steps
                - 'affine_bounds': Skip affine bounds computation and subsequent steps
                - 'initial_v_bounds': Skip initial value vector bounds computation and subsequent steps
                - 'tightened_v_bounds': Skip tightened value vector bounds computation and subsequent steps
        """
        self._check_all_vars_set()

        if use_decomp:
            # Step 1: ML bounds
            if self.skip_from_step != 'ml_bounds':
                ml_bounds = self._compute_ml_bounds(verbose=verbose)
                self._set_ml_bounds(ml_bounds)
            else:
                return self._optimize(sense=sense, verbose=verbose, gurobi_params=gurobi_params)

            # Step 2: Parameter bounds from affine relationships
            if self.skip_from_step != 'affine_bounds':
                affine_bounds = self._compute_affine_bounds(ml_bounds)
                self._set_affine_bounds(affine_bounds)
            else:
                return self._optimize(sense=sense, verbose=verbose, gurobi_params=gurobi_params)

            # Step 3: Initial value vector bounds
            if self.skip_from_step != 'initial_v_bounds':
                initial_v_bounds = self._compute_initial_v_bounds()
                self._set_initial_v_bounds(initial_v_bounds)
            else:
                return self._optimize(sense=sense, verbose=verbose, gurobi_params=gurobi_params)

            # Step 4: Tightened value vector bounds
            if self.skip_from_step != 'tightened_v_bounds':
                tightened_v_bounds = self._compute_tightened_v_bounds()
                self._set_tightened_v_bounds(tightened_v_bounds)

        return self._optimize(sense=sense, verbose=verbose, gurobi_params=gurobi_params)

def generate_regression_data(n_samples=10000, n_features=10, sigma=1.0, beta=None, random_seed=None):
    """
    Generates a synthetic linear regression dataset:
      y = X * beta + noise.

    Arguments:
    - n_samples: number of samples
    - n_features: number of features
    - sigma: standard deviation of Gaussian noise
    - beta: optional; if provided, must be a 1D array of length d
    - random_seed: for reproducibility

    Returns:
    - X: feature matrix of shape (n, d)
    - y: response vector of shape (n,)
    - beta: the true coefficients used to generate y
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Generate feature matrix X
    X = np.random.randn(n_samples, n_features)

    # If no beta provided, draw a random one
    if beta is None:
        beta = np.random.randn(n_features)

    # Generate noise
    noise = sigma * np.random.randn(n_samples)

    # Generate response
    y = X @ beta + noise

    return X, y

def generate_classification_data(n_samples=10000, n_features=10, beta=None, random_seed=None):
    """
    Generates a synthetic binary classification dataset using a logistic model:
      p = sigmoid(X * beta)
      y ~ Bernoulli(p)

    Arguments:
    - n_samples: number of samples
    - n_features: number of features
    - beta: optional; if provided, must be a 1D array of length d
    - random_seed: for reproducibility

    Returns:
    - X: feature matrix of shape (n, d)
    - y: binary labels in {0, 1} of shape (n,)
    - beta: the true coefficients used to generate y
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Generate feature matrix X
    X = np.random.randn(n_samples, n_features)

    # If no beta provided, draw a random one
    if beta is None:
        beta = np.random.randn(n_features)

    # Compute probabilities via logistic function
    logits = X @ beta
    p = 1 / (1 + np.exp(-logits))

    # Sample y from Bernoulli(p)
    y = np.random.binomial(n=1, p=p)

    return X, y

def generate_random_data(n_samples=10000, n_features=5, n_states=10):
    """Generate random training data for one reward and one probability model"""
    return generate_regression_data(n_samples=n_samples, n_features=n_features),\
          generate_classification_data(n_samples=n_samples, n_features=n_features)

def setup_and_solve_mrp(n_states, reward_model, pi_model, p_models, method='direct', ablation_step=None, verbose=False, gurobi_logfile=None):
    """Setup and solve MRP with given models
    p_models: list of models for P matrix rows (remaining rows will be uniform)"""

    # Create MRP object
    mrp = MarkovReward(n_states=n_states, n_features=5) if ablation_step is None\
         else AblatedMarkov(n_states=n_states, n_features=5, skip_from_step=ablation_step)

    # Add models
    mrp.add_ml_model(reward_model)
    mrp.add_ml_model(pi_model)
    for model in p_models:
        mrp.add_ml_model(model)

    # Set rewards (bounded between -1 and 1)
    r_1 = mrp.ml_outputs[0][0]
    rewards = [r_1 / (i+1) for i in range(n_states)]
    mrp.set_r(rewards)

    # Set initial distribution with careful balancing
    pi_n = mrp.ml_outputs[1][0]
    pi = [pi_n, 1 - pi_n] + [0] * (n_states - 2)  # Rest are 0
    mrp.set_pi(pi)

    # Set P matrix with absorbing state
    p_rows = []
    n_p_models = len(p_models)

    # First n_p_models rows use ML models
    for i in range(n_p_models):
        p_n = mrp.ml_outputs[i+2][0]  # Probability of staying in current state
        row = [0] * n_states
        current_state = i % (n_states-1)
        next_state = (current_state + 1) % (n_states-1)

        # Model predicts probability of staying, remainder goes to next state
        row[current_state] = p_n
        row[next_state] = 1 - p_n
        p_rows.append(row)

    # Remaining rows use uniform transitions
    for i in range(n_states - n_p_models - 1):
        row = [1.0/n_states] * n_states
        p_rows.append(row)

    # Last row is absorbing state
    p_rows.append([0] * (n_states-1) + [1])

    mrp.set_P(p_rows)

    # Set tighter feature bounds
    for i in range(mrp.n_features):
        mrp.features[i].LB = -1
        mrp.features[i].UB = 1

    with open(gurobi_logfile, 'a') if gurobi_logfile else nullcontext() as logfile:
        with redirect_stdout(logfile) if gurobi_logfile else nullcontext():
            start_time = time()
            gurobi_params = {
                'TimeLimit': 1200,
                'LogFile': gurobi_logfile if gurobi_logfile else ''
            }

            if method == 'direct':
                result = mrp.optimize(use_decomp=False, verbose=verbose,
                                   gurobi_params=gurobi_params)
            else:
                result = mrp.optimize(use_decomp=True, verbose=verbose,
                                   gurobi_params=gurobi_params)
            solve_time = time() - start_time

    return {
        'status': result['status'],
        'time': solve_time,
        'objective': result.get('objective', None),
        'message': result.get('message', None)
    }

def run_state_scaling(n_states_list=[5, 10, 20, 50, 100, 200], n_instances=10, verbose=False, gurobi_logfile=None):
    """Experiment 1: State Space Scaling"""
    results = []

    for n_states in n_states_list:
        print(f"** Running experiments for n_states = {n_states} **")

        for instance in range(n_instances):
            print(f"+ Running instance {instance} +")
            # Generate data for three models
            reg_data, clf_data = generate_random_data(n_states=n_states)

            # Train simple models
            reward_model = LinearRegression().fit(*reg_data)
            pi_model = LogisticRegression().fit(*clf_data)
            p_model = LogisticRegression().fit(*clf_data)  # One row of P

            # Solve with both methods
            for method in ['direct', 'decomp']:
                print(f"\tSolving with {method} method")
                result = setup_and_solve_mrp(
                    n_states=n_states,
                    reward_model=reward_model,
                    pi_model=pi_model,
                    p_models=[p_model],  # Single P model
                    method=method,
                    verbose=verbose,
                    gurobi_logfile=gurobi_logfile
                )

                print(f"\tSolved in {result['time']:.2f} seconds" +
                      (f" objective {result['objective']:.2f}" if result['objective'] is not None else '') +
                      (f" with {result['status']} status" if result['status'] is not None else '') +
                      (f" ({result['message']})" if result['message'] is not None else ''))

                results.append({
                    'n_states': n_states,
                    'instance': instance,
                    'method': method,
                    **result
                })

    return pd.DataFrame(results)

def run_model_scaling(n_instances=10, n_states=20, verbose=False, gurobi_logfile=None):
    """Experiment 2: Model Count Scaling"""
    results = []

    for n_p_models in range(1, n_states):  # 1 to n_states P models (last state is absorbing)
        print(f"** Running experiments for {n_p_models + 2} total models **")

        for instance in range(n_instances):
            print(f"+ Running instance {instance} +")
            # Generate data
            reg_data, clf_data = generate_random_data(n_states=n_states)

            # Train models
            reward_model = LinearRegression().fit(*reg_data)
            pi_model = LogisticRegression().fit(*clf_data)
            p_models = [
                LogisticRegression().fit(*clf_data)
                for _ in range(n_p_models)
            ]

            # Solve with both methods
            for method in ['direct', 'decomp']:
                print(f"\tSolving with {method} method")
                result = setup_and_solve_mrp(
                    n_states=n_states,
                    reward_model=reward_model,
                    pi_model=pi_model,
                    p_models=p_models,
                    method=method,
                    verbose=verbose,
                    gurobi_logfile=gurobi_logfile
                )

                print(f"\tSolved in {result['time']:.2f} seconds" +
                      (f" objective {result['objective']:.2f}" if result['objective'] is not None else '') +
                      (f" with {result['status']} status" if result['status'] is not None else '') +
                      (f" ({result['message']})" if result['message'] is not None else ''))

                results.append({
                    'n_models': n_p_models + 2,  # Add 2 for reward and pi models
                    'instance': instance,
                    'method': method,
                    **result
                })

    return pd.DataFrame(results)

def run_tree_depth_scaling(depths=[2, 3, 4, 6, 8], n_instances=10, n_states=20, verbose=False, gurobi_logfile=None):
    """Experiment 3: Decision Tree Depth Scaling"""
    results = []


    for depth in depths:
        print(f"** Running experiments for depth = {depth} **")

        for instance in range(n_instances):
            print(f"+ Running instance {instance} +")
            # Generate data
            reg_data, clf_data = generate_random_data(n_states=n_states)

            # Train decision trees with same depth
            reward_model = DecisionTreeRegressor(max_depth=depth).fit(*reg_data)
            pi_model = DecisionTreeClassifier(max_depth=depth).fit(*clf_data)
            p_model = DecisionTreeClassifier(max_depth=depth).fit(*clf_data)

            # Solve with both methods
            for method in ['direct', 'decomp']:
                print(f"\tSolving with {method} method")
                result = setup_and_solve_mrp(
                    n_states=n_states,
                    reward_model=reward_model,
                    pi_model=pi_model,
                    p_models=[p_model],  # One P model
                    method=method,
                    verbose=verbose,
                    gurobi_logfile=gurobi_logfile
                )

                print(f"\tSolved in {result['time']:.2f} seconds" +
                      (f" objective {result['objective']:.2f}" if result['objective'] is not None else '') +
                      (f" with {result['status']} status" if result['status'] is not None else '') +
                      (f" ({result['message']})" if result['message'] is not None else ''))

                results.append({
                    'depth': depth,
                    'instance': instance,
                    'method': method,
                    **result
                })

    return pd.DataFrame(results)

def run_nn_scaling(n_instances=10, n_states=20, verbose=False, gurobi_logfile=None):
    """Experiment 4: Neural Network Architecture Scaling"""
    results = []


    # Configurations
    n_layers_list = [1, 2]
    n_neurons_list = [5, 10, 15, 20]

    for n_layers in n_layers_list:
        for n_neurons in n_neurons_list:
            print(f"** Running experiments for {n_layers} layers with {n_neurons} neurons each **")

            for instance in range(n_instances):
                print(f"+ Running instance {instance} +")
                # Generate data
                reg_data, clf_data = generate_random_data(n_states=n_states)

                # Configure architecture
                hidden_layers = tuple([n_neurons] * n_layers)

                # Train neural networks with same architecture
                reward_model = MLPRegressor(
                    hidden_layer_sizes=hidden_layers,
                    activation='relu',
                    max_iter=1000,
                    early_stopping=True
                ).fit(*reg_data)

                pi_model = MLPClassifier(
                    hidden_layer_sizes=hidden_layers,
                    activation='relu',
                    max_iter=1000,
                    early_stopping=True
                ).fit(*clf_data)

                p_model = MLPClassifier(
                    hidden_layer_sizes=hidden_layers,
                    activation='relu',
                    max_iter=1000,
                    early_stopping=True
                ).fit(*clf_data)

                # Solve with both methods
                for method in ['direct', 'decomp']:
                    print(f"\tSolving with {method} method")
                    result = setup_and_solve_mrp(
                        n_states=n_states,
                        reward_model=reward_model,
                        pi_model=pi_model,
                        p_models=[p_model],  # One P model
                        method=method
                    )

                    print(f"\tSolved in {result['time']:.2f} seconds" +
                          (f" objective {result['objective']:.2f}" if result['objective'] is not None else '') +
                          (f" with {result['status']} status" if result['status'] is not None else '') +
                          (f" ({result['message']})" if result['message'] is not None else ''))

                    results.append({
                        'n_layers': n_layers,
                        'n_neurons': n_neurons,
                        'instance': instance,
                        'method': method,
                        **result
                    })

    return pd.DataFrame(results)

def run_ablation(n_instances=10, verbose=False, gurobi_logfile=None):
    """Experiment 5: Ablation Studies"""
    results = []
    n_states = 20

    # Define ablation configurations to test
    ablation_steps = [
        'ml_bounds',  # skip from ML bounds onwards
        'affine_bounds',  # skip from affine bounds onwards
        'initial_v_bounds',  # skip from initial v bounds onwards
        'tightened_v_bounds',  # skip from tightened v bounds onwards
        None # baseline
    ]

    # generate instances
    instances = []
    for i in range(n_instances):
        reg_data, clf_data = generate_random_data(n_states=n_states)

        # Train models
        reward_model = DecisionTreeRegressor(max_depth=8).fit(*reg_data)
        pi_model = DecisionTreeClassifier(max_depth=8).fit(*clf_data)
        p_model = DecisionTreeClassifier(max_depth=8).fit(*clf_data)

        instances.append((reward_model, pi_model, p_model))

    # Then test each ablation on all instances
    for instance_idx, (reward_model, pi_model, p_model) in enumerate(instances):
        print(f"\n** Testing instance {instance_idx + 1}/{n_instances} **")

        for skip_from_step in ablation_steps:
            step_name = skip_from_step if skip_from_step else 'baseline'
            print(f"\tRunning ablation from {step_name}")

            # Solve using existing setup function
            result = setup_and_solve_mrp(
                n_states=n_states,
                reward_model=reward_model,
                pi_model=pi_model,
                p_models=[p_model],
                method='decomp',
                ablation_step=skip_from_step,
                verbose=verbose,
                gurobi_logfile=gurobi_logfile
            )

            print(f"\tSolved in {result['time']:.2f} seconds" +
                  (f" objective {result['objective']:.2f}" if result['objective'] is not None else '') +
                  (f" with {result['status']} status" if result['status'] is not None else '') +
                  (f" ({result['message']})" if result['message'] is not None else ''))

            results.append({
                'ablation_step': step_name,
                'instance': instance_idx,
                **result
            })

    return pd.DataFrame(results)

# TODO: TEST THIS ON LAPTOP
def run_small_experiment(verbose=False, gurobi_logfile=None):
    """Run a small version of each experiment for testing"""
    results = {}

    print("\n=== Running Small Experiments ===")

    # Small State Scaling (just 3 and 4 states, 2 instances each)
    print("\nRunning Small State Scaling...")
    results['states'] = run_state_scaling(
        n_states_list=[3, 4],
        n_instances=2,
        verbose=verbose,
        gurobi_logfile=gurobi_logfile
    )

    # Small Model Count (3 states, 3-5 models, 2 instances each)
    print("\nRunning Small Model Count Scaling...")
    results['models'] = run_model_scaling(
        n_instances=2,
        n_states=3,
        verbose=verbose,
        gurobi_logfile=gurobi_logfile
    )

    # Small Tree Depth (3 states, depths 2 and 3, 2 instances each)
    print("\nRunning Small Tree Depth Scaling...")
    results['trees'] = run_tree_depth_scaling(
        depths=[2, 3],
        n_instances=2,
        n_states=3,
        verbose=verbose,
        gurobi_logfile=gurobi_logfile
    )

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run MarkovML experiments')

    # Experiment selection arguments
    parser.add_argument('--all', action='store_true', help='Run all experiments')
    parser.add_argument('--states', action='store_true', help='Run state scaling experiment')
    parser.add_argument('--models', action='store_true', help='Run model count scaling experiment')
    parser.add_argument('--trees', action='store_true', help='Run decision tree depth scaling experiment')
    parser.add_argument('--nns', action='store_true', help='Run neural network architecture scaling experiment')
    parser.add_argument('--ablation', action='store_true', help='Run ablation studies')
    parser.add_argument('--small', action='store_true', help='Run small version of experiments')

    # Global parameters
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--logfile', type=str, help='Path to Gurobi log file (default: None)')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Directory to store results (default: results)')

    args = parser.parse_args()

    # If no experiments specified, show help
    if not any([args.all, args.states, args.models, args.trees, args.nns, args.ablation, args.small]):
        parser.print_help()
        exit(1)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Create Gurobi log file path if specified
    gurobi_logfile = os.path.join(args.output_dir, args.logfile) if args.logfile else None

    results = {}

    # Run requested experiments
    if args.small:
        print("\nRunning small version of experiments...")
        results = run_small_experiment(verbose=args.verbose, gurobi_logfile=gurobi_logfile)
        results['states'].to_csv(os.path.join(args.output_dir, 'results_state_scaling_small.csv'))
        results['models'].to_csv(os.path.join(args.output_dir, 'results_model_scaling_small.csv'))
        results['trees'].to_csv(os.path.join(args.output_dir, 'results_tree_depth_small.csv'))
    else:
        if args.all or args.states:
            print("\nRunning State Scaling Experiment...")
            results['states'] = run_state_scaling(
                verbose=args.verbose,
                gurobi_logfile=gurobi_logfile
            )
            results['states'].to_csv(os.path.join(args.output_dir, 'results_state_scaling.csv'))

        if args.all or args.models:
            print("\nRunning Model Count Scaling Experiment...")
            results['models'] = run_model_scaling(
                verbose=args.verbose,
                gurobi_logfile=gurobi_logfile
            )
            results['models'].to_csv(os.path.join(args.output_dir, 'results_model_scaling.csv'))

        if args.all or args.trees:
            print("\nRunning Decision Tree Depth Scaling Experiment...")
            results['trees'] = run_tree_depth_scaling(
                verbose=args.verbose,
                gurobi_logfile=gurobi_logfile
            )
            results['trees'].to_csv(os.path.join(args.output_dir, 'results_tree_depth.csv'))

        if args.all or args.nns:
            print("\nRunning Neural Network Architecture Scaling Experiment...")
            results['nns'] = run_nn_scaling(
                verbose=args.verbose,
                gurobi_logfile=gurobi_logfile
            )
            results['nns'].to_csv(os.path.join(args.output_dir, 'results_nn_scaling.csv'))

        if args.all or args.ablation:
            print("\nRunning Ablation Experiment...")
            results['ablation'] = run_ablation(
                verbose=args.verbose,
                gurobi_logfile=gurobi_logfile
            )
            results['ablation'].to_csv(os.path.join(args.output_dir, 'results_ablation.csv'))

    # Print summary statistics for completed experiments
    print("\n=== Summary Statistics ===")
    for name, df in results.items():
        print(f"\n=== {name.title()} Experiment Summary ===")
        # Filter for optimal solutions and get solve times
        optimal_df = df[df['status'] == 'optimal']

        if name == 'ablation':
            grouped = optimal_df.groupby('ablation_step')['time']
        else:
            grouped = optimal_df.groupby('method')['time']

        # Calculate statistics for each group
        summary = grouped.agg([
            ('geometric_mean', lambda x: np.exp(np.mean(np.log(x)))),
            ('geometric_std', lambda x: np.exp(np.std(np.log(x)))),
            ('min', 'min'),
            ('max', 'max')
        ])
        print(summary.round(3))

import numpy as np
import gurobipy as gp
from gurobipy import GRB
from gurobi_ml.sklearn import (add_linear_regression_constr,
                               add_logistic_regression_constr,
                               add_decision_tree_regressor_constr,
                               add_random_forest_regressor_constr,
                               add_mlp_regressor_constr)
from gurobi_ml.torch import add_sequential_constr
from markovml.utils.gurobi_ml_ext import (add_decision_tree_classifier_constr,
                           add_random_forest_classifier_constr,
                           add_mlp_classifier_constr,
                           add_decision_rules_constr,
                           add_sequential_classifier_constr,
                           _get_sequential_n_outputs)
from typing import Optional, List, Dict, Any, Sequence, Tuple
from abc import ABC, abstractmethod
import re
from sklearn.linear_model import (LinearRegression,
                                  LogisticRegression,
                                  Ridge,
                                  Lasso)
from sklearn.ensemble import (RandomForestRegressor,
                              RandomForestClassifier)
from sklearn.neural_network import (MLPRegressor, MLPClassifier)
from sklearn.tree import (DecisionTreeRegressor, DecisionTreeClassifier)
from markovml.utils.models_ext import DecisionRules, SequentialClassifier
from torch.nn import Sequential
from time import time
import markovml.utils.ima as ima

class AbstractMarkov(ABC):
    """
    This is an abstract class for Markov processes with ML models embedded.
    It implements all the basic functionality for setting up a Markov process,
    adding ML models, defining the feature space, and optimizing, including
    with the decomposition and bound propagation scheme from the paper.
    It will be subclassed by `MarkovReward`, `MarkovReach`, and `MarkovHitting`,
    which specialize the functionality to the specific problems.
    """

    # small number for strict inequalities
    _epsilon = 1e-6

    # Supported ml models
    # Currently supported:
    #   linear regression
    #   ridge regression
    #   lasso regression
    #   logistic regression
    #   decision tree regression
    #   decision tree classifier
    #   random forest regression
    #   random forest classifier
    #   mlp regression
    #   mlp classifier
    #   torch sequential
    #   decision rules
    _ml_model_registry = {
        (LinearRegression, Ridge, Lasso): {
            'constraint_method': add_linear_regression_constr,
            'get_n_outputs': lambda model: 1,
            'bounds': (-GRB.INFINITY, GRB.INFINITY)
        },
        LogisticRegression: {
            'constraint_method': add_logistic_regression_constr,
            'get_n_outputs': lambda model: 1,
            'bounds': (0, 1)
        },
        DecisionTreeRegressor: {
            'constraint_method': add_decision_tree_regressor_constr,
            'bounds': (-GRB.INFINITY, GRB.INFINITY),
            'get_n_outputs': lambda model: model.n_outputs_
        },
        DecisionTreeClassifier: {
            'constraint_method': add_decision_tree_classifier_constr,
            'bounds': (0, 1),
            'get_n_outputs': lambda model: model.n_outputs_
        },
        RandomForestRegressor: {
            'constraint_method': add_random_forest_regressor_constr,
            'bounds': (-GRB.INFINITY, GRB.INFINITY),
            'get_n_outputs': lambda model: model.n_outputs_
        },
        RandomForestClassifier: {
            'constraint_method': add_random_forest_classifier_constr,
            'bounds': (0, 1),
            'get_n_outputs': lambda model: model.n_outputs_
        },
        MLPRegressor: {
            'constraint_method': add_mlp_regressor_constr,
            'bounds': (-GRB.INFINITY, GRB.INFINITY),
            'get_n_outputs': lambda model: model.n_outputs_
        },
        MLPClassifier: {
            'constraint_method': add_mlp_classifier_constr,
            'bounds': (0, 1),
            'get_n_outputs': lambda model: model.n_outputs_
        },
        DecisionRules: {
            'constraint_method': add_decision_rules_constr,
            'bounds': (-GRB.INFINITY, GRB.INFINITY),
            'get_n_outputs': lambda model: 1
        },
        SequentialClassifier: {
            'constraint_method': add_sequential_classifier_constr,
            'bounds': (0, 1),
            'get_n_outputs': _get_sequential_n_outputs
        },
        Sequential: {
            'constraint_method': add_sequential_constr,
            'bounds': (-GRB.INFINITY, GRB.INFINITY),
            'get_n_outputs': _get_sequential_n_outputs
        }
    }

    def __init__(self, n_states: int, n_features: int):
        """
        Initialize the Markov process with the given number of states and features.

        Parameters
        ----------
        n_states : int
            Number of states in the Markov process (`n` in the paper)
        n_features : int
            Number of features in the Markov process (`m` in the paper)

        Returns
        -------
        None

        Notes
        -----
        This initializes the basic attributes of the Markov process including:
        - Number of states and features
        - Gurobi optimization model
        - Parameter dictionaries for variables, ML output flags, and dimensions
        - Feature space variables and constraints
        - ML model and output storage

        Subclasses will have additional attributes in their constructors.
        """
        # basic attributes
        self.n_states = n_states
        self.n_features = n_features

        # create the main optimization model
        self.model = gp.Model("mainmodel")

        # store the paramaters
        self.variables: Dict[str, gp.Var] = {}
        self._var_is_ml_output: Dict[str, bool] = {}
        self._var_is_const: Dict[str, bool] = {}
        self._var_dim: Dict[str, int] = {}

        # create the feature space
        self.features = self.model.addVars(
            n_features,
            name="features",
            lb=-GRB.INFINITY,
            ub=GRB.INFINITY,
            vtype=GRB.CONTINUOUS
        )
        self.feature_aux = []
        self.feature_constraints = []

        # store the ml models
        self.ml_models = []
        self.ml_outputs = []

    # METHODS THAT MUST BE IMPLEMENTED BY SUBCLASSES

    @abstractmethod
    def _initialize_parameters_and_constraints(self):
        """
        Here, we add the specific parameters needed for that subclass,
        set the appropriate flags for whether they are set, specify their
        dimensions, and set them to values if provided. As well, here we
        must add necessary constraints for the subclass, for example,
        the definition of the value vector v.
        """
        pass

    @abstractmethod
    def _check_all_vars_set(self):
        """
        Checks which variables have to be set, either to ML outputs or constants,
        i.e., the variables which must be bound via affine equalities, before
        calling optimize.
        """
        pass

    @abstractmethod
    def _set_objective(self):
        """
        Set the optimization objective.
        """
        pass

    @abstractmethod
    def _set_feasible_constraints(self, lb, ub):
        """
        Set the constraints on the appropriate quantity for the feasibility version.
        The constraints are named "temp_feasible_lb" and "temp_feasible_ub" so that
        they can be removed after the optimization.
        """
        pass

    @abstractmethod
    def get_values(self):
        """
        Get values of the key variables of the Markov process after optimization.
        """
        pass

    @abstractmethod
    def _compute_affine_bounds(self, ml_bounds):
        """
        Propagate the bounds on the ML outputs to the parameters. Must be
        specified by the subclass which parameters are bound by affine equalities
        to ML outputs. It should return a dictionary containing bounds for each
        parameter that is an ML output.
        """
        pass

    @abstractmethod
    def _set_affine_bounds(self, affine_bounds):
        """
        Set the bounds on the parameters based on the affine bounds.
        """
        pass

    @abstractmethod
    def _compute_initial_v_bounds(self):
        """
        Compute initial bounds on the value vector v. Depends on the subclass.
        """
        pass

    @abstractmethod
    def _set_initial_v_bounds(self, initial_v_bounds):
        """
        Set the initial bounds on the value vector v.
        """
        pass

    @abstractmethod
    def _compute_tightened_v_bounds(self):
        """
        Use Gauss-Seidel to compute tighter bounds on value vector.
        Must be implemented by subclasses based on their linear system.
        """
        pass

    @abstractmethod
    def _set_tightened_v_bounds(self, tightened_v_bounds):
        """
        Set the tightened bounds on the value vector v.
        """
        pass

    # FUNCTIONS FOR CHECKING VALIDITY OF INPUTTED PARAMETERS

    def _validate_vector_length(self, values: Sequence, var_name: str):
        """
        Validate the length of a vector.

        Parameters
        ----------
        values : Sequence
            The vector whose length needs to be validated
        var_name : str
            Name of the variable for error messages

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the length of values does not match the expected length stored in
            self._var_dim[var_name][0]

        See Also
        --------
        _validate_matrix_shape
        """
        expected_length = self._var_dim[var_name][0]
        if len(values) != expected_length:
            raise ValueError(f"{var_name} must have length {expected_length}")

    def _validate_matrix_shape(self, values: Sequence[Sequence], var_name: str):
        """
        Validate the shape of a matrix.

        Parameters
        ----------
        values : Sequence[Sequence]
            The matrix whose shape needs to be validated
        var_name : str
            Name of the variable for error messages


        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the number of rows in values does not match the expected number of rows stored in
            self._var_dim[var_name][0]

        See Also
        --------
        _validate_vector_length
        """
        expected_rows, expected_cols = self._var_dim[var_name]
        if len(values) != expected_rows:
            raise ValueError(f"{var_name} must have {expected_rows} rows")
        if not all(isinstance(row, Sequence) for row in values):
            raise ValueError(f"{var_name} must be a 2D sequence (matrix) of size {expected_rows}x{expected_cols}")
        if any(len(row) != expected_cols for row in values):
            raise ValueError(f"Each row of {var_name} must have length {expected_cols}")

    def _validate_vector_bounds(self, values: Sequence, var_name: str, bounds: Tuple[float, float]):
        """
        Validate the bounds of a vector.

        Parameters
        ----------
        values : Sequence
            The vector whose bounds need to be validated
        var_name : str
            Name of the variable for error messages
        bounds : Tuple[float, float]
            The bounds to validate against

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If any value in values does not satisfy the bounds

        See Also
        --------
        _validate_matrix_bounds
        """
        if not all(bounds[0] <= val <= bounds[1] for val in values):
            raise ValueError(f"{var_name} must contain values between {bounds[0]} and {bounds[1]}")

    def _validate_matrix_bounds(self, values: Sequence[Sequence], var_name: str, bounds: Tuple[float, float]):
        """
        Validate the bounds of a matrix.

        Parameters
        ----------
        values : Sequence[Sequence]
            The matrix whose bounds need to be validated
        var_name : str
            Name of the variable for error messages
        bounds : Tuple[float, float]
            The bounds to validate against

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If any value in values does not satisfy the bounds

        See Also
        --------
        _validate_vector_bounds
        """
        for i, row in enumerate(values):
            if not all(bounds[0] <= val <= bounds[1] for val in row):
                raise ValueError(f"Row {i} of {var_name} must contain values between {bounds[0]} and {bounds[1]}")

    def _validate_stochastic_vector(self, values: Sequence, var_name: str, sub=False, strictsub=False):
        """
        Validate probability vector.

        Parameters
        ----------
        values : Sequence
            The vector of probabilities to validate
        var_name : str
            Name of the variable for error messages
        sub : bool
            If True, allow substochastic (sum <= 1)
        strictsub : bool
            If True, require strict substochastic (sum < 1)

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the sum of the probabilities does not satisfy the given condition

        See Also
        --------
        _validate_stochastic_matrix
        """
        sum_values = sum(values)

        if strictsub:
            if not sum_values < 1:
                raise ValueError(f"The sum of probabilities in {var_name} must be strictly less than 1 (got {sum_values})")
        elif sub:
            if not sum_values <= 1:
                raise ValueError(f"The sum of probabilities in {var_name} must be at most 1 (got {sum_values})")
        else:
            if not abs(sum_values - 1) < 1e-10:  # Use small tolerance for floating point
                raise ValueError(f"The sum of probabilities in {var_name} must be 1 (got {sum_values})")

    def _validate_stochastic_matrix(self, values: Sequence[Sequence], var_name: str, sub=False, strictsub=False):
        """
        Validate a stochastic matrix.

        Parameters
        ----------
        values : Sequence[Sequence]
            The matrix of probabilities to validate
        var_name : str
            Name of the variable for error messages
        sub : bool
            If True, allow substochastic (sum <= 1)
        strictsub : bool
            If True, require strict substochastic (sum < 1)

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the sum of the probabilities does not satisfy the given condition

        See Also
        --------
        _validate_stochastic_vector
        """
        for i, row in enumerate(values):
            self._validate_stochastic_vector(row, f"{var_name}[{i}]", sub=sub, strictsub=strictsub)

    # FUNCTIONS FOR CHECKING THE SETTING OF PARAMETERS

    def _check_any_var_set(self, var_name: str):
        """
        Check if any elements of the variable are already set (either to ML output or constant).

        Parameters
        ----------
        var_name : str
            Name of the variable to check

        Returns
        -------
        bool
            True if any elements are set, False if none are set

        Notes
        -----
        Can be used in setting functions to make sure that they have not already been set.
        """
        if len(self._var_dim[var_name]) == 1:
            return any(self._var_is_ml_output[var_name]) or any(self._var_is_const[var_name])
        else:
            return any(item for sublist in self._var_is_ml_output[var_name] for item in sublist) or \
                any(item for sublist in self._var_is_const[var_name] for item in sublist)

    # FUNCTIONS TO SET PARAMETERS
    # We can set them to a constant value, or to an ML output
    # Can set them individually with set_to_const or set_to_ml_output
    # Checks if already set to an ML output or a constant
    # In subclasses, may have helper functions to e.g., set all of pi at once

    def set_to_ml_output(self, var : gp.Var, expr : gp.LinExpr | gp.Var):
        """
        Set a variable to an ML output. Checks if already has been set to an ML output or a constant.

        Parameters
        ----------
        var : gp.Var
            The variable to set
        expr : gp.LinExpr | gp.Var
            The expression to set the variable to

        Returns
        -------
        None

        Notes
        -----
        In practice, introduces an equality constraint to the model.

        Raises
        ------
        ValueError
            If unknown variable type
            If the variable is already set to an ML output or a constant
        """
        var_name = var.VarName
        base_name = var_name.split('[')[0]  # Get base name (pi, P, etc.)

        if base_name not in self._var_is_ml_output:
            raise ValueError(f"Unknown variable type: {base_name}")

        indices = self._parse_var_indices(var_name)
        if isinstance(indices, int):
            i = indices
            if self._var_is_ml_output[base_name][i] or self._var_is_const[base_name][i]:
                raise ValueError(f"{var_name} is already set to an ML output or constant")
            self._var_is_ml_output[base_name][i] = True
        elif len(indices) == 2:
            i, j = indices
            if self._var_is_ml_output[base_name][i][j] or self._var_is_const[base_name][i][j]:
                raise ValueError(f"{var_name} is already set to an ML output or constant")
            self._var_is_ml_output[base_name][i][j] = True

        self.model.addConstr(var == expr, name=f"set_to_ml_output_{var_name}")
        self.model.update()

    def set_to_const(self, var : gp.Var, value : float | int):
        """
        Set a variable to a constant value. Checks if already has been set to a constant or an ML output.

        Parameters
        ----------
        var : gp.Var
            The variable to set
        value : float | int
            The constant value to set the variable to

        Returns
        -------
        None

        Notes
        -----
        In practice, introduces an equality constraint to the model and fixes
        upper and lower bounds of the variable to the given value.

        Raises
        ------
        ValueError
            If unknown variable type
            If the variable is already set to a constant or an ML output
        """

        var_name = var.VarName
        base_name = var_name.split('[')[0]  # Get base name (pi, P, etc.)

        if base_name not in self._var_is_const:
            raise ValueError(f"Unknown variable type: {base_name}")

        indices = self._parse_var_indices(var_name)
        if isinstance(indices, int):
            i = indices
            if self._var_is_const[base_name][i] or self._var_is_ml_output[base_name][i]:
                raise ValueError(f"{var_name} is already set to a constant or ML output")
            self._var_is_const[base_name][i] = True
        elif len(indices) == 2:
            i, j = indices
            if self._var_is_const[base_name][i][j] or self._var_is_ml_output[base_name][i][j]:
                raise ValueError(f"{var_name} is already set to a constant or ML output")
            self._var_is_const[base_name][i][j] = True

        self.model.addConstr(var == value, name=f"set_to_const_{var_name}")
        var.LB = var.UB = value
        self.model.update()

    # FUNCTIONS TO SPECIFY THE FEATURE SPACE
    # The feature space is a MILP-representable space of features
    # We add linear constraints
    # Or, we can add an auxiliary variable, to help us represent integral constraints
    # This feature space is copied over when optimizing over ML outputs
    # Theoretically, these two building blocks can represent any MILP-representable feature space

    def add_feature_constraint(self, expr : gp.LinExpr):
        """
        Add a feature constraint to the model.

        Parameters
        ----------
        expr : gp.LinExpr
            The linear expression to add as a constraint

        Returns
        -------
        None

        Notes
        -----
        Adds the constraint to the model and also stores it in self.feature_constraints.
        Constraints that are added here are also copied over when optimizing over ML outputs.
        """
        self.feature_constraints.append(self.model.addConstr(expr))
        self.model.update()

    def add_feature_aux_variable(self, *args, **kwargs):
        """
        Add a feature auxiliary variable to the model.

        Parameters
        ----------
        *args : tuple
            Arguments to pass to self.model.addVar
        **kwargs : dict
            Keyword arguments to pass to self.model.addVar

        Returns
        -------
        gp.Var
            The auxiliary variable added to the model

        Notes
        -----
        Adds a variable to the model and stores it in self.feature_aux. Variables
        added here are also copied over when optimizing over ML outputs. Arguments
        are passed to Gurobi's addVar function, so can specify variable types, bounds,
        etc. Hence, can encode arbitrary MILP constraints by adding integer auxiliary
        variables.
        """
        var = self.model.addVar(*args, **kwargs)
        self.feature_aux.append(var)
        self.model.update()
        return var

    # FUNCTIONS TO ADD CONSTRAINTS TO THE MRP
    # For example, inequalities on pi, P, r
    # They are only added to the overall model, not to the submodels

    def add_constraint(self, constr : gp.Constr):
        """
        Add an inequality to the overall model.
        Could just call self.model.addConstr(constr), but this is a wrapper for clarity.

        Parameters
        ----------
        constr : gp.Constr
            The constraint to add

        Returns
        -------
        None
        """
        self.model.addConstr(constr)
        self.model.update()

    # FUNCTIONS TO ADD THE ML MODELS
    # Can add ML models to the MRP, which creates variables for the ML outputs
    # These can then be used to specify the params
    # To see supported models, see the _ml_model_registry dictionary

    def add_ml_model(self, ml_model : Any):
        """
        Add an ML model to the Markov process. Computes the constraints linking
        the ML outputs to the features, and creates variables for the ML outputs.
        ML outputs can then be accessed using self.ml_outputs.

        Parameters
        ----------
        ml_model : Any
            The ML model to add

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the ML model type is not supported
        """
        # Find matching model type in registry
        for model_types, config in self._ml_model_registry.items():
            if isinstance(ml_model, model_types):
                self.ml_models.append(ml_model)

                # Create output variables based on model type
                n_outputs = config['get_n_outputs'](ml_model)
                lb, ub = config['bounds']
                self.ml_outputs.append(
                    self.model.addVars(
                        n_outputs,
                        name=f"ml_outputs_{len(self.ml_outputs)}",
                        lb=lb,
                        ub=ub,
                        vtype=GRB.CONTINUOUS
                    )
                )

                # Add constraints using registered method
                config['constraint_method'](
                    self.model,
                    ml_model,
                    self.features,
                    self.ml_outputs[-1]
                )

                self.model.update()
                return

        raise ValueError(f"Unsupported ML model type: {type(ml_model)}")

    # FUNCTIONS TO OPTIMIZE THE PROBLEM

    def optimize(self, use_decomp : bool = True, sense : str = "max", verbose : bool = False, gurobi_params : dict = None):
        """
        Public interface for optimizing the problem.
        If use_decomp is True, we use the decomposition and bound propagation approach.
        Can also set the sense, verbosity, and additional Gurobi parameters if needed.

        Parameters
        ----------
        use_decomp : bool (default=True)
            Whether to use the decomposition and bound propagation approach
        sense : str (default="max")
            The sense of the optimization problem, either "max" or "min"
        verbose : bool (default=False)
            Whether to print Gurobi output
        gurobi_params : dict (default=None)
            Additional Gurobi parameters to set, e.g., {'TimeLimit': 60}

        Returns
        -------
        dict
            A dictionary containing the optimization result. Possible formats:

            For optimal/suboptimal solutions:
            {
                'status': str, either 'optimal' or 'suboptimal',
                'objective': float, the objective value,
                'values': dict, the values of the variables
            }

            For infeasible/unbounded problems:
            {
                'status': str, either 'infeasible' or 'unbounded',
                'message': str, description of the problem
            }

            For other errors:
            {
                'status': 'error',
                'message': str, error description with status code
            }

        Notes
        -----
        First checks if all required variables are set, then implements the decomposition
        and bound propagation approach, and lastly solves the bilinear program.
        """
        # Check if all required variables are set
        self._check_all_vars_set()

        if use_decomp:
            # Step 1: ML bounds
            ml_bounds = self._compute_ml_bounds(verbose=verbose)
            self._set_ml_bounds(ml_bounds)

            # Step 2: Parameter bounds from affine relationships
            affine_bounds = self._compute_affine_bounds(ml_bounds)
            self._set_affine_bounds(affine_bounds)

            # Step 3: Initial value vector bounds
            initial_v_bounds = self._compute_initial_v_bounds()
            self._set_initial_v_bounds(initial_v_bounds)

            # Step 4: Tightened value vector bounds
            tightened_v_bounds = self._compute_tightened_v_bounds()
            self._set_tightened_v_bounds(tightened_v_bounds)

        return self._optimize(sense=sense, verbose=verbose, gurobi_params=gurobi_params)

    def find_feasible(self, lb : float | int | None = None, ub : float | int | None = None, use_decomp : bool = True, verbose : bool = False, gurobi_params : dict = None):
        """
        Public interface for finding a feasible solution to the problem.
        It finds a feasible feature vector.

        Parameters
        ----------
        lb : float | int | None (default=None)
            The lower bound on the feature vector. None means -inf.
        ub : float | int | None (default=None)
            The upper bound on the feature vector. None means inf.
        use_decomp : bool (default=True)
            Whether to use the decomposition and bound propagation approach
        verbose : bool (default=False)
            Whether to print Gurobi output
        gurobi_params : dict (default=None)
            Additional Gurobi parameters to set, e.g., {'TimeLimit': 60}

        Returns

        dict
            A dictionary containing the feasibility result. Possible formats:

            For feasible solutions:
            {
                'status': 'feasible',
                'values': dict, the values of the variables
            }

            For infeasible/unbounded problems:
            {
                'status': str, either 'infeasible' or 'unbounded',
                'message': str, description of the problem
            }

            For other errors:
            {
                'status': 'error',
                'message': str, error description with status code
            }

        """
        self._check_all_vars_set()

        if lb is None and ub is None:
            raise ValueError("Must specify at least one bound")
        if lb is not None and ub is not None:
            if lb > ub:
                raise ValueError("Lower bound must be less than upper bound")

        if use_decomp:
            # Step 1: ML bounds
            ml_bounds = self._compute_ml_bounds(verbose=verbose)
            self._set_ml_bounds(ml_bounds)

            # Step 2: Parameter bounds from affine relationships
            affine_bounds = self._compute_affine_bounds(ml_bounds)
            self._set_affine_bounds(affine_bounds)

            # Step 3: Initial value vector bounds
            initial_v_bounds = self._compute_initial_v_bounds()
            self._set_initial_v_bounds(initial_v_bounds)

            # Step 4: Tightened value vector bounds
            tightened_v_bounds = self._compute_tightened_v_bounds()
            self._set_tightened_v_bounds(tightened_v_bounds)

        return self._find_feasible(lb, ub, verbose=verbose, gurobi_params=gurobi_params)

    def _optimize(self, sense : str = "max", verbose : bool = False, gurobi_params : dict = None):
        """
        Private interface for optimizing the problem.
        """
        # Set the objective (implemented by subclasses)
        self._set_objective()

        # Common Gurobi parameter setting
        self.model.setParam("OutputFlag", verbose)
        if gurobi_params is not None:
            for param, value in gurobi_params.items():
                self.model.setParam(param, value)
        self.model.setParam("Presolve", 0)
        self.model.ModelSense = GRB.MAXIMIZE if sense == "max" else GRB.MINIMIZE

        self.model.optimize()

        # Check optimization status
        if self.model.status == GRB.OPTIMAL:
            return {
                'status': 'optimal',
                'objective': self.model.objVal,
                'values': self.get_values()
            }
        elif self.model.status == GRB.SUBOPTIMAL:
            return {
                'status': 'suboptimal',
                'objective': self.model.objVal,
                'values': self.get_values()
            }
        elif self.model.status == GRB.INFEASIBLE:
            return {
                'status': 'infeasible',
                'message': 'Problem is infeasible'
            }
        elif self.model.status == GRB.UNBOUNDED:
            return {
                'status': 'unbounded',
                'message': 'Problem is unbounded'
            }
        else:
            return {
                'status': 'error',
                'message': f'Optimization failed with status code {self.model.status}'
            }

    def _find_feasible(self, lb : float | int | None = None, ub : float | int | None = None, verbose : bool = False, gurobi_params : dict = None):
        """
        Private interface for finding a feasible solution to the problem.
        """
        if lb is None:
            lb = -GRB.INFINITY
        if ub is None:
            ub = GRB.INFINITY
        # must be implemented by subclasses
        self._set_feasible_constraints(lb, ub)

        # Common Gurobi parameter setting
        self.model.setParam("OutputFlag", verbose)
        if gurobi_params is not None:
            for param, value in gurobi_params.items():
                self.model.setParam(param, value)
        self.model.setObjective(0)
        self.model.setParam("Presolve", 0)

        self.model.optimize()

        status = self.model.status
        if status == GRB.OPTIMAL:
            values = self.get_values()

        # Remove temporary constraints after storing status and values
        for constr in self.model.getQConstrs():
            if constr.QCName == "temp_feasible_lb" or constr.QCName == "temp_feasible_ub":
                self.model.remove(constr)
        self.model.update()

        if status == GRB.OPTIMAL:
            return {
                'status': 'feasible',
                'values': values
            }
        elif status == GRB.INFEASIBLE:
            return {
                'status': 'infeasible',
                'message': 'Problem is infeasible'
            }
        elif status == GRB.UNBOUNDED:
            return {
                'status': 'unbounded',
                'message': 'Problem is unbounded'
            }
        else:
            return {
                'status': 'error',
                'message': f'Optimization failed with status code {status}'
            }

    def get_value(self, var : gp.Var | str):
        """
        Get value of variable after optimization.
        Can either pass in a Gurobi variable or a string (name of variable).

        Parameters
        ----------
        var : gp.Var | str
            The variable to get the value of. If a string, it is the name of the variable.

        Returns
        -------
        float
            The value of the variable
        """
        if isinstance(var, gp.Var):
            return var.X  # Direct access if it's already a Gurobi variable
        else:
            return self.model.getVarByName(str(var)).X  # Use name lookup for strings

    # DECOMPOSITION AND BOUND PROPAGATION FUNCTIONS
    # Here, ML models can be optimized

    def _optimize_ml_output(self, ml_model_index : int = 0, ml_output_index : int = 0, sense : str = "max", verbose : bool = False):
        """
        Minimize the specified ML output over the constraints defined for self.features and self.feature_aux.

        Parameters
        ----------
        ml_model_index : int (default=0)
            Index of the ML model to optimize
        ml_output_index : int (default=0)
            Index of the ML output to minimize
        sense : str (default="max")
            The sense of the optimization problem, either "max" or "min"
        verbose : bool (default=False)
            Whether to print Gurobi output

        Returns
        -------
        tuple
            A tuple containing the status of the optimization and the value of the ML output

        Notes
        -----
        Builds the subproblem for the given ML model and output, and solves it.
        If the optimizer returns an optimal solution, it returns the objective value.
        If the optimizer returns a suboptimal solution, it uses the best known objective bound.
        If the optimizer returns an infeasible or unbounded solution, it returns None. This is later
        detected in _set_ml_bounds, which forces the bounds to be conflicting, so that the overall model
        is easily detected as infeasible.
        """
        # Create a new subproblem model
        self.model.update()
        subproblem = gp.Model("opt_ml_output")

        # Add feature variables to the subproblem with the same bounds as the original model
        sub_features = {}
        for var in self.features.values():
            sub_features[var.VarName] = subproblem.addVar(
                name=var.VarName,
                lb=var.LB,  # Copy the lower bound
                ub=var.UB,  # Copy the upper bound
                vtype=var.VType  # Copy the variable type (e.g., CONTINUOUS)
            )
        # Add auxiliary feature variables to the subproblem
        sub_features_aux = {}
        for var in self.feature_aux:
            sub_features_aux[var.VarName] = subproblem.addVar(
                name=var.VarName,
                lb=var.LB,  # Copy the lower bound
                ub=var.UB,  # Copy the upper bound
                vtype=var.VType  # Copy the variable type (e.g., CONTINUOUS)
            )
        subproblem.update()

        # Convert self.features to a dict mapping VarName to GRB.Var for easier access
        features_map = {var.VarName: var for var in self.features.values()}
        features_aux_map = {var.VarName: var for var in self.feature_aux}

        # Iterate over constraints in the original model
        for constr in self.feature_constraints:
            # Get the linear expression for the constraint
            expr = self.model.getRow(constr)

            new_expr = gp.LinExpr()
            for i in range(expr.size()):
                var = expr.getVar(i)
                coeff = expr.getCoeff(i)
                if var.VarName in features_map:
                    new_expr += coeff * sub_features[var.VarName]
                elif var.VarName in features_aux_map:
                    new_expr += coeff * sub_features_aux[var.VarName]

            # Add the modified constraint to the subproblem
            subproblem.addLConstr(new_expr, constr.Sense, constr.RHS, name=constr.ConstrName)
            subproblem.update()

        # Add the ML output variable to the subproblem
        ml_output = subproblem.addVars(
            len(self.ml_outputs[ml_model_index]),
            name=f"ml_output_{ml_output_index}",
            lb=self.ml_outputs[ml_model_index][ml_output_index].LB,
            ub=self.ml_outputs[ml_model_index][ml_output_index].UB,
            vtype=self.ml_outputs[ml_model_index][ml_output_index].VType
        )

        # Recreate the ML model constraint in the subproblem
        # Find matching model type in registry
        ml_model = self.ml_models[ml_model_index]
        for model_types, config in self._ml_model_registry.items():
            if isinstance(ml_model, model_types):
                config['constraint_method'](
                    subproblem,
                    ml_model,
                    sub_features,
                    ml_output
                )
                break
        else:
            raise ValueError(f"Unsupported ML model type for optimization: {type(ml_model)}")

        # Set the objective to optimize the ML output
        sense = GRB.MAXIMIZE if sense == "max" else GRB.MINIMIZE
        subproblem.setObjective(ml_output[ml_output_index], sense)
        subproblem.setParam("OutputFlag", verbose)
        subproblem.setParam("Presolve", 0)

        # Solve the subproblem
        subproblem.optimize()

        if subproblem.status == GRB.OPTIMAL:
            return ('optimal', ml_output[ml_output_index].X)
        elif subproblem.status in [GRB.SUBOPTIMAL, GRB.TIME_LIMIT, GRB.NUMERIC]:
            # Use ObjBound which is guaranteed to be a valid bound
            return ('suboptimal', subproblem.ObjBound)
        elif subproblem.status == GRB.INFEASIBLE:
            return ('infeasible', None)
        elif subproblem.status == GRB.UNBOUNDED:
            return ('unbounded', float('-inf') if sense == "min" else float('inf'))
        else:
            # For any other status, if we have a bound, use it
            if hasattr(subproblem, 'ObjBound'):
                return ('error', subproblem.ObjBound)
            # Otherwise use conservative bounds
            return ('error', float('-inf') if sense == "min" else float('inf'))

    def _compute_ml_bounds(self, verbose : bool = False):
        """
        Get the bounds on all ML outputs.

        Parameters
        ----------
        verbose : bool (default=False)
            Whether to print Gurobi output

        Returns
        -------
        list[list[tuple[float | None, float | None]]]
            A list of lists of (max, min) tuples for each ML model and output

        Notes
        -----
        If `verbose` is True, it also prints the bounds for each ML model and output, or
        infeasible status. The `verbose` flag is also passed to Gurobi.
        """
        # get bounds on ml outputs
        ml_bounds = []
        for i in range(len(self.ml_models)):
            ml_bounds.append([])
            for j in range(len(self.ml_outputs[i])):
                min_status, min_val = self._optimize_ml_output(i, j, sense="min", verbose=verbose)
                max_status, max_val = self._optimize_ml_output(i, j, sense="max", verbose=verbose)

                if min_status == 'infeasible' or max_status == 'infeasible':
                    ml_bounds[i].append((None, None))
                    if verbose:
                        print(f"ML model {i} output {j} is infeasible")
                else:
                    ml_bounds[i].append((min_val, max_val))
                    if verbose:
                        print(f"ML model {i} output {j} bounds: {min_val}, {max_val}")
        return ml_bounds

    def _set_ml_bounds(self, ml_bounds: list[list[tuple[float | None, float | None]]]):
        """
        Set the bounds on the ML outputs.

        Parameters
        ----------
        ml_bounds : list[list[tuple[float | None, float | None]]]
            A list of lists of (max, min) tuples for each ML model and output

        Notes
        -----
        If a tuple is (None, None), this means that the ML model was
        infeasible, so here we set the bounds to conflicting values,
        so that the overall model is easily detected as infeasible.
        """
        for i in range(len(self.ml_models)):
            for j in range(len(self.ml_outputs[i])):
                lb, ub = ml_bounds[i][j]
                if lb is None or ub is None:
                    self.ml_outputs[i][j].LB = GRB.INFINITY
                    self.ml_outputs[i][j].UB = -GRB.INFINITY
                else:
                    self._update_lb(self.ml_outputs[i][j], lb)
                    self._update_ub(self.ml_outputs[i][j], ub)

    # SOME HELPER FUNCTIONS

    def _update_lb(self, var : gp.Var, lb : float | int):
        """
        Update the lower bound of a variable if the new bound is greater than the current bound.
        Includes a check to ensure that the new bound is not greater than the upper bound.

        Parameters
        ----------
        var : gp.Var
            The variable to update the lower bound of
        lb : float | int
            The new lower bound

        Returns
        -------
        None
        """
        if lb > var.LB:
            if lb > var.UB:
                var.LB = var.UB
            else:
                var.LB = lb
            self.model.update()

    def _update_ub(self, var : gp.Var, ub : float | int):
        """
        Update the upper bound of a variable if the new bound is less than the current bound.
        Includes a check to ensure that the new bound is not less than the lower bound.

        Parameters
        ----------
        var : gp.Var
            The variable to update the upper bound of
        ub : float | int
            The new upper bound

        Returns
        -------
        None
        """
        if ub < var.UB:
            if ub < var.LB:
                var.UB = var.LB
            else:
                var.UB = ub
            self.model.update()

    def _compute_expr_bounds(self, expr : gp.LinExpr, bounds_dict : dict):
        """
        Compute bounds for an affine expression given bounds on its variables.

        Parameters
        ----------
        expr : gp.LinExpr
            The affine expression to compute bounds for
        bounds_dict : dict
            A dictionary mapping variable names to (min, max) bounds

        Returns
        -------
        tuple
            A tuple containing the lower and upper bounds of the expression

        Notes
        -----
        Just simple check of the coefficients and their signs, and sums up appropriate bounds.
        Used to propagate bounds on ML outputs to the the parameters of the Markov process.
        """
        lb = 0
        ub = 0

        # Handle constant term
        if hasattr(expr, 'getConstant'):
            const = expr.getConstant()
            lb += const
            ub += const

        # Handle variables
        for i in range(expr.size()):
            coeff = expr.getCoeff(i)
            var = expr.getVar(i)
            var_bounds = bounds_dict[var.VarName]

            # If coefficient is positive, use min for LB and max for UB
            if coeff > 0:
                lb += coeff * var_bounds[0]
                ub += coeff * var_bounds[1]
            # If coefficient is negative, use max for LB and min for UB
            else:
                lb += coeff * var_bounds[1]
                ub += coeff * var_bounds[0]

        return lb, ub

    def _get_ml_expr(self, var : gp.Var):
        """
        Gets the affine expression for a parameter set to an ML output. Isolates the variable.
        This is used to propagate bounds on ML outputs to the the parameters of the Markov process.

        Parameters
        ----------
        var : gp.Var
            The variable to get the ML expression for

        Returns
        -------
        gp.LinExpr
            The linear expression for the ML output
        """
        constr = self.model.getConstrByName(f"set_to_ml_output_{var.VarName}")
        expr = self.model.getRow(constr)  # Gets the full expression (pi[0] + x[0] - 1 == 0)
        rhs = constr.RHS
        # Collect terms with and without our variable
        var_coeff = 0
        other_terms = gp.LinExpr()
        const = expr.getConstant()

        for i in range(expr.size()):
            coeff = expr.getCoeff(i)
            term_var = expr.getVar(i)
            if term_var.VarName == var.VarName:
                var_coeff = coeff
            else:
                other_terms += coeff * term_var

        # Rearrange to isolate var: var = (rhs - other_terms - const) / var_coeff
        if var_coeff == 0:
            raise ValueError(f"Could not find variable {var.VarName} in its ML constraint")
        return (rhs - other_terms - const) / var_coeff

    def _parse_var_indices(self, var_name: str):
        """
        Parse indices from Gurobi variable names like 'pi[0]', 'r[1]', or 'P[0,1]'
        Returns int for single indices and tuple for multiple indices.

        Parameters
        ----------
        var_name : str
            The name of the Gurobi variable (e.g., 'pi[0]', 'P[0,1]')

        Returns
        -------
        int | tuple
            Single index (e.g., 0 for 'pi[0]') or tuple of indices (e.g., (0,1) for 'P[0,1]')

        Raises
        ------
        ValueError
            If the variable name cannot be parsed
        """
        pattern = r'\[(\d+)(?:,(\d+))?\]'
        match = re.search(pattern, var_name)

        if not match:
            raise ValueError(f"Could not parse indices from variable name: {var_name}")

        # Filter out None values and convert to integers
        indices = tuple(int(idx) for idx in match.groups() if idx is not None)

        # Return single integer if only one index, otherwise return tuple
        return indices[0] if len(indices) == 1 else indices

    def _seq_is_all_const(self, seq : Sequence[Any] | Sequence[Sequence[Any]]):
        """
        Check if all elements in sequence (1D or 2D) are constants.
        Specifically, checks if they are all floats or ints.

        Parameters
        ----------
        seq : Sequence[Any] | Sequence[Sequence[Any]]
            The sequence to check

        Returns
        -------
        bool
            True if all elements in the sequence are constants, False otherwise
        """
        if not seq:
            return True

        # If first element is a sequence (except string), treat as 2D
        if isinstance(seq[0], (list, tuple, np.ndarray)) and not isinstance(seq[0], str):
            return all(isinstance(value, (float, int))
                    for row in seq
                    for value in row)
        # Otherwise treat as 1D
        else:
            return all(isinstance(value, (float, int))
                    for value in seq)

class MarkovReward(AbstractMarkov):
    """
    For the total reward problem.
    """
    def __init__(self, n_states: int,
                 n_features: int,
                 discount_factor: float = 0.97, # 3% discount
                 pi: Optional[List | np.ndarray] = None,
                 P: Optional[List[List] | np.ndarray] = None,
                 r: Optional[List | np.ndarray] = None):
        """
        Initialize a Markov reward process for the total reward problem.

        Parameters
        ----------
        n_states : int
            The number of states
        n_features : int
            The number of features
        discount_factor : float (default=0.97)
            The discount factor for the Markov process
        pi : Optional[List | np.ndarray] (default=None)
            The initial state probabilities
        P : Optional[List[List] | np.ndarray] (default=None)
            The transition probabilities
        r : Optional[List | np.ndarray] (default=None)
            The reward vector

        Notes
        -----
        Calls the constructor of the parent class AbstractMarkov, and additionally stores
        the discount factor. Then, initializes the defining parameters and constraints.
        """

        super().__init__(n_states, n_features)
        self.discount_factor = discount_factor
        self._initialize_parameters_and_constraints(pi, P, r)

    # FUNCTIONS THAT MUST BE IMPLEMENTED IN SUBCLASSES WHICH UNIQUELY DEFINE THE PROBLEM

    def _initialize_parameters_and_constraints(self,
                                               pi : Optional[List | np.ndarray] = None,
                                               P : Optional[List[List] | np.ndarray] = None,
                                               r : Optional[List | np.ndarray] = None):
        """
        Initializes the parameters and constraints for a Markov reward process.

        Parameters
        ----------
        pi : Optional[List | np.ndarray] (default=None)
            The initial state probabilities
        P : Optional[List[List] | np.ndarray] (default=None)
            The transition probabilities
        r : Optional[List | np.ndarray] (default=None)
            The reward vector

        Returns
        -------
        None

        Notes
        -----
        Adds variables for: pi (bounds between 0 and 1), P (bounds between 0 and 1),
        r (bounds between -inf and inf), and v (bounds between -inf and inf).
        Creates boolean lists to store whether each variable has been set to an ML output
        or a constant. Adds stochastic constraints on pi and rows of P. Lastly, adds
        the Bellman equation constraint: v = r + discount_factor * P * v.
        """
        self.variables['pi'] = self.model.addVars(
            self.n_states,
            name="pi",
            lb=0,
            ub=1,
            vtype=GRB.CONTINUOUS
        )

        self.variables['P'] = self.model.addVars(
            self.n_states,
            self.n_states,
            name="P",
            lb=0,
            ub=1,
            vtype=GRB.CONTINUOUS
        )

        self.variables['r'] = self.model.addVars(
            self.n_states,
            name="r",
            lb=-GRB.INFINITY,
            ub=GRB.INFINITY,
            vtype=GRB.CONTINUOUS
        )

        self.variables['v'] = self.model.addVars(
            self.n_states,
            name="v",
            lb=-GRB.INFINITY,
            ub=GRB.INFINITY,
            vtype=GRB.CONTINUOUS
        )

        self.model.update()

        self._var_is_ml_output['pi'] = [False for _ in range(self.n_states)]
        self._var_is_const['pi'] = [False for _ in range(self.n_states)]
        self._var_is_ml_output['P'] = [[False for _ in range(self.n_states)] for _ in range(self.n_states)]
        self._var_is_const['P'] = [[False for _ in range(self.n_states)] for _ in range(self.n_states)]
        self._var_is_ml_output['r'] = [False for _ in range(self.n_states)]
        self._var_is_const['r'] = [False for _ in range(self.n_states)]
        self._var_is_ml_output['v'] = [False for _ in range(self.n_states)]
        self._var_is_const['v'] = [False for _ in range(self.n_states)]

        self._var_dim['pi'] = (self.n_states,)
        self._var_dim['P'] = (self.n_states, self.n_states)
        self._var_dim['r'] = (self.n_states,)
        self._var_dim['v'] = (self.n_states,)

        if pi is not None:
            self.set_pi(pi)
        if P is not None:
            self.set_P(P)
        if r is not None:
            self.set_r(r)

        # add probability constraints
        self.model.addConstr(gp.quicksum(self.variables['pi'][i] for i in range(self.n_states)) == 1)
        for i in range(self.n_states):
            self.model.addConstr(gp.quicksum(self.variables['P'][i, j] for j in range(self.n_states)) == 1,
                                 name=f"prob_constr_{i}")

        # add Bellman equation constraints
        for i in range(self.n_states):
            self.model.addConstr(self.variables['v'][i] == self.variables['r'][i] + self.discount_factor * gp.quicksum(self.variables['P'][i, j] * self.variables['v'][j] for j in range(self.n_states)),
                                 name=f"bellman_constr_{i}")

        self.model.update()

    def _check_all_vars_set(self):
        """
        Check if pi, P, and r have been set before optimization.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If any required variables have not been set
        """
        for i in range(self._var_dim['pi'][0]):
            if not (self._var_is_ml_output['pi'][i] or self._var_is_const['pi'][i]):
                raise ValueError(f"pi[{i}] needs to be set to a constant or linked to an ML output")
        for i in range(self._var_dim['r'][0]):
            if not (self._var_is_ml_output['r'][i] or self._var_is_const['r'][i]):
                raise ValueError(f"r[{i}] needs to be set to a constant or linked to an ML output")
        for i in range(self._var_dim['P'][0]):
            for j in range(self._var_dim['P'][1]):
                if not (self._var_is_ml_output['P'][i][j] or self._var_is_const['P'][i][j]):
                    raise ValueError(f"P[{i}, {j}] needs to be set to a constant or linked to an ML output")

    def _set_objective(self):
        """
        Sets the objective function to pi * v.

        Returns
        -------
        None
        """
        self.model.setObjective(gp.quicksum(self.pi[i] * self.v[i] for i in range(self.n_states)))

    def _set_feasible_constraints(self, lb : float | int, ub : float | int):
        """
        Sets the feasible constraints on the objective. When looking for a feasible solution,
        this simply adds the constraints: lb <= pi * v <= ub.

        Parameters
        ----------
        lb : float | int
            The lower bound
        ub : float | int
            The upper bound

        Returns
        -------
        None
        """
        self.model.addConstr(gp.quicksum(self.pi[i] * self.v[i] for i in range(self.n_states)) >= lb, name="temp_feasible_lb")
        self.model.addConstr(gp.quicksum(self.pi[i] * self.v[i] for i in range(self.n_states)) <= ub, name="temp_feasible_ub")

    def get_values(self):
        """
        Returns the values of pi, P, r, v, features, and ml_outputs as a dictionary.

        Returns
        -------
        dict
            A dictionary with keys:
            - `pi`: the values of pi
            - `P`: the values of P
            - `r`: the values of r
            - `v`: the values of v
            - `features`: the values of the features
            - `ml_outputs`: the values of the ml_outputs
        """
        return {
            "pi": [self.pi[i].x for i in range(self.n_states)],
            "P": [[self.P[i, j].x for j in range(self.n_states)] for i in range(self.n_states)],
            "r": [self.r[i].x for i in range(self.n_states)],
            "v": [self.v[i].x for i in range(self.n_states)],
            "features": [self.features[i].x for i in range(self.n_features)],
            "ml_outputs": [[self.ml_outputs[i][j].x for j in range(len(self.ml_outputs[i]))] for i in range(len(self.ml_outputs))]
        }

    def _compute_affine_bounds(self, ml_bounds : List[List[Tuple[float, float]]]):
        """
        Computes the affine bounds for the parameters that are linked to ML outputs.

        Parameters
        ----------
        ml_bounds : List[List[Tuple[float, float]]]
            The bounds for the ML outputs

        Returns
        -------
        dict
            A dictionary with keys: 'pi', 'P', 'r'
        """
        # Convert ml_bounds to dictionary for easier lookup
        bounds_dict = {}
        for i, model_bounds in enumerate(ml_bounds):
            for j, (lb, ub) in enumerate(model_bounds):
                bounds_dict[f"ml_outputs_{i}[{j}]"] = (lb, ub)

        # Initialize bounds dictionary
        affine_bounds = {
            'pi': [],
            'P': [],
            'r': []
        }

        # Compute bounds for each parameter that is an ML output
        for i in range(self._var_dim['pi'][0]):
            if self._var_is_ml_output['pi'][i]:
                expr = self._get_ml_expr(self.pi[i])
                affine_bounds['pi'].append((i, self._compute_expr_bounds(expr, bounds_dict)))

        for i in range(self._var_dim['r'][0]):
            if self._var_is_ml_output['r'][i]:
                expr = self._get_ml_expr(self.r[i])
                affine_bounds['r'].append((i, self._compute_expr_bounds(expr, bounds_dict)))

        for i in range(self._var_dim['P'][0]):
            for j in range(self._var_dim['P'][1]):
                if self._var_is_ml_output['P'][i][j]:
                    expr = self._get_ml_expr(self.P[i, j])
                    affine_bounds['P'].append((i, j, self._compute_expr_bounds(expr, bounds_dict)))

        return affine_bounds

    def _set_affine_bounds(self, affine_bounds : dict):
        """
        Sets the bounds for the parameters that are linked to ML outputs.

        Parameters
        ----------
        affine_bounds : dict
            The bounds for the ML outputs

        Returns
        -------
        None
        """
        for i, (lb, ub) in affine_bounds['pi']:
            self._update_lb(self.pi[i], lb)
            self._update_ub(self.pi[i], ub)

        for i, (lb, ub) in affine_bounds['r']:
            self._update_lb(self.r[i], lb)
            self._update_ub(self.r[i], ub)

        for i, j, (lb, ub) in affine_bounds['P']:
            self._update_lb(self.P[i, j], lb)
            self._update_ub(self.P[i, j], ub)

    def _compute_initial_v_bounds(self):
        """
        Computes the initial bounds for v using the reward bounds and discount factor.
        Uses the lemma from the paper: for each $v_i$, $v_i \\in [v^min_i, v^max_i]$ where
        $v^min_i = \\min_j r^min_j / (1 - \\lambda)$, and
        $v^max_i = \\max_j r^max_j / (1 - \\lambda)$.

        Returns
        -------
        List[Tuple[float, float]]
            The initial bounds for v
        """
        # Get min/max rewards across all states
        r_min = min(self.r[i].LB for i in range(self._var_dim['r'][0]))
        r_max = max(self.r[i].UB for i in range(self._var_dim['r'][0]))

        # Compute v bounds using reward bounds and discount factor
        v_min = r_min / (1 - self.discount_factor)
        v_max = r_max / (1 - self.discount_factor)

        # Return bounds for each state
        return [(v_min, v_max) for _ in range(self._var_dim['v'][0])]

    def _set_initial_v_bounds(self, initial_v_bounds):
        """
        Sets the initial bounds for v.

        Parameters
        ----------
        initial_v_bounds : List[Tuple[float, float]]
            The initial bounds for v

        Returns
        -------
        None
        """
        for i in range(self._var_dim['v'][0]):
            lb, ub = initial_v_bounds[i]
            self._update_lb(self.v[i], lb)
            self._update_ub(self.v[i], ub)

    def _compute_tightened_v_bounds(self):
        """
        Computes the tightened bounds for v using the bounds for P and r and initial bounds for v.
        This uses Gauss-Seidel to solve the linear system: $vM = r$, where $M = I - \\lambda * P$.

        Returns
        -------
        List[Tuple[float, float]]
            The tightened bounds for v
        """
        # Create interval matrices for P and r
        P_intervals = np.zeros((self._var_dim['P'][0], self._var_dim['P'][1], 2))
        r_intervals = np.zeros((self._var_dim['r'][0], 2))

        # Fill intervals from current bounds
        for i in range(self._var_dim['P'][0]):
            for j in range(self._var_dim['P'][1]):
                P_intervals[i,j] = [self.P[i,j].LB, self.P[i,j].UB]
            r_intervals[i] = [self.r[i].LB, self.r[i].UB]

        # Create linear system: vM = r, where M = I - discount_factor * P
        I = ima._identity_interval(self._var_dim['v'][0])
        M = ima.subtract(I, self.discount_factor * P_intervals)

        # Initial guess using current v bounds
        v0 = np.array([[self.v[i].LB, self.v[i].UB] for i in range(self._var_dim['v'][0])])

        # Solve using Gauss-Seidel
        v_intervals = ima.interval_gauss_seidel(
            M, r_intervals, v0,
            max_iter=100,
            tol=1e-6
        )

        # Convert to list of tuples
        return [(v_intervals[i][0], v_intervals[i][1]) for i in range(self._var_dim['v'][0])]

    def _set_tightened_v_bounds(self, tightened_v_bounds):
        """
        Sets the tightened bounds for v.

        Parameters
        ----------
        tightened_v_bounds : List[Tuple[float, float]]
            The tightened bounds for v

        Returns
        -------
        None
        """
        for i in range(self._var_dim['v'][0]):
            lb, ub = tightened_v_bounds[i]
            self._update_lb(self.v[i], lb)
            self._update_ub(self.v[i], ub)

    # CLEAN PARAMETER ACCESSORS

    @property
    def pi(self):
        """
        Accessor for the pi parameter.

        Returns
        -------
        gurobipy.tupledict
            The pi parameter
        """
        return self.variables['pi']

    @property
    def P(self):
        """
        Accessor for the P parameter.

        Returns
        -------
        gurobipy.tupledict
            The P parameter
        """
        return self.variables['P']

    @property
    def r(self):
        """
        Accessor for the r parameter.

        Returns
        -------
        gurobipy.tupledict
            The r parameter
        """
        return self.variables['r']

    @property
    def v(self):
        """
        Accessor for the v parameter.

        Returns
        -------
        gurobipy.tupledict
            The v parameter
        """
        return self.variables['v']

    # HELPER FUNCTIONS TO SET PARAMETERS
    # Can set them all at once with set_pi, set_P, set_r
    # These check for validity of, e.g., probabilities

    def set_pi(self, values: List[float | int | gp.Var | gp.LinExpr]):
        """
        Helper function to set the entire pi vector.

        Parameters
        ----------
        values : List[float | int | gp.Var | gp.LinExpr]
            The values to set the pi parameter to

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the values have incorrect length, are not in [0, 1], don't sum to 1,or pi is already set

        Notes
        -----
        This checks for validity of the values, checks if pi is already set, and sets the values.
        """
        self._validate_vector_length(values, "pi")
        if self._seq_is_all_const(values):
            self._validate_vector_bounds(values, "pi", (0, 1))
            self._validate_stochastic_vector(values, "pi")
        assert not self._check_any_var_set("pi"),\
              "pi is already set"

        for i in range(self._var_dim["pi"][0]):
            if isinstance(values[i], (float, int)):
                self.set_to_const(self.pi[i], values[i])
            elif isinstance(values[i], (gp.Var, gp.LinExpr)):
                self.set_to_ml_output(self.pi[i], values[i])
            else:
                raise ValueError(f"pi[{i}] must be a float, int, or a Gurobi expression")

    def set_r(self, values: List[float | int | gp.Var | gp.LinExpr]):
        """
        Helper function to set the entire r vector.

        Parameters
        ----------
        values : List[float | int | gp.Var | gp.LinExpr]
            The values to set the r parameter to

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the values have incorrect length, or r is already set

        Notes
        -----
        This checks for validity of the values, checks if r is already set, and sets the values.
        """
        self._validate_vector_length(values, "r")
        assert not self._check_any_var_set("r"),\
              "r is already set"

        for i in range(self._var_dim["r"][0]):
            if isinstance(values[i], (float, int)):
                self.set_to_const(self.r[i], values[i])
            elif isinstance(values[i], (gp.Var, gp.LinExpr)):
                self.set_to_ml_output(self.r[i], values[i])
            else:
                raise ValueError(f"r[{i}] must be a float, int, or a Gurobi expression")

    def set_P(self, values: List[List[float | int | gp.Var | gp.LinExpr]]):
        """
        Helper function to set the entire P matrix.

        Parameters
        ----------
        values : List[List[float | int | gp.Var | gp.LinExpr]]
            The values to set the P parameter to

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the values have incorrect shape, are not in [0, 1], rows don't sum to 1, or P is already set

        Notes
        -----
        This checks for validity of the values, checks if P is already set, and sets the values.
        """
        self._validate_matrix_shape(values, "P")
        if self._seq_is_all_const(values):
            self._validate_matrix_bounds(values, "P", (0, 1))
            self._validate_stochastic_matrix(values, "P")
        for i in range(len(values)):
            if self._seq_is_all_const(values[i]):
                self._validate_stochastic_vector(values[i], f"P[{i}]")

        assert not self._check_any_var_set("P"),\
              "P is already set"

        for i in range(self._var_dim["P"][0]):
            for j in range(self._var_dim["P"][1]):
                if isinstance(values[i][j], (float, int)):
                    self.set_to_const(self.P[i, j], values[i][j])
                elif isinstance(values[i][j], (gp.Var, gp.LinExpr)):
                    self.set_to_ml_output(self.P[i, j], values[i][j])
                else:
                    raise ValueError(f"P[{i}, {j}] must be a float, int, or a Gurobi expression")

    # FUNCTIONS TO ADD CONSTRAINTS TO THE MRP
    # In an MRP, IFT is important because we have a full prob matrix

    def add_ifr_inequalities(self):
        """
        Helper function to add IFR (increasing failure rate) inequalities on P.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Notes
        -----
        The definition of IFR is the following. Given an $n \\times n$ transition matrix $P$, and a fixed $i \\in [n]$,
        define the function $b_h(i) = \\sum_{j=h}^n P_{ij}$. Then, $P$ is IFR if, for each $h \\in [n]$, we have
        $b_h(i) \\leq b_h(i+1)$ for all $i \\in [n-1]$.
        """
        for h in range(self.n_states):
            for i in range(self.n_states - 1):
                # b_h(i) <= b_h(i+1), i.e., sum_{j=h}^n P_{ij} <= sum_{j=h}^n P_{i+1, j}
                self.model.addConstr(gp.quicksum(self.P[i, j] for j in range(h, self.n_states)) <=
                                    gp.quicksum(self.P[i + 1, j] for j in range(h, self.n_states)),
                                    name=f"ifr_h_{h}_row_{i}_row_{i+1}")

        self.model.update()

class MarkovReach(AbstractMarkov):
    """
    For the reachability problem.
    """
    def __init__(self, n_states: int,
                 n_features: int,
                 n_targets: int,
                 n_transient: int,
                 pi: Optional[List | np.ndarray] = None,
                 Q: Optional[List[List] | np.ndarray] = None,
                 R: Optional[List[List] | np.ndarray] = None):
        """
        Initialize a Markov chain for the reachability problem.

        Parameters
        ----------
        n_states : int
            The number of states
        n_features : int
            The number of features
        n_targets : int
            The number of target states ($|S|$ in the paper)
        n_transient : int
            The number of transient states ($|T|$ in the paper)
        pi : Optional[List | np.ndarray]
            The initial state distribution over transient states
        Q : Optional[List[List] | np.ndarray]
            The transition matrix between transient states
        R : Optional[List[List] | np.ndarray]
            The transition matrix from transient states to target states

        Raises
        -------
        ValueError
            If the number of target states is greater than or equal to the number of states, or the number of transient
            states is greater than or equal to the number of states

        Notes
        -----
        Calls the constructor of the parent class AbstractMarkov, with additional attributes.
        Then, initializes the defining parameters and constraints for the reachability problem.
        """

        if n_targets >= n_states:
            raise ValueError("Number of target states must be less than number of states")
        if n_transient >= n_states:
            raise ValueError("Number of transient states must be less than number of states")

        super().__init__(n_states, n_features)
        self.n_targets = n_targets
        self.n_transient = n_transient
        self._initialize_parameters_and_constraints(pi, Q, R)

    # FUNCTIONS THAT MUST BE IMPLEMENTED IN SUBCLASSES WHICH UNIQUELY DEFINE THE PROBLEM

    def _initialize_parameters_and_constraints(self, pi : Optional[List | np.ndarray] = None,
                                               Q : Optional[List[List] | np.ndarray] = None,
                                               R : Optional[List[List] | np.ndarray] = None):
        """
        Initializes the parameters and constraints for the reachability problem.

        Parameters
        ----------
        pi : Optional[List | np.ndarray]
            The initial state distribution over transient states
        Q : Optional[List[List] | np.ndarray]
            The transition matrix between transient states
        R : Optional[List[List] | np.ndarray]
            The transition matrix from transient states to target states

        Returns
        -------
        None

        Notes
        -----
        Adds variables for: pi (bounds between 0 and 1), Q (bounds between 0 and 1, strictly less
        than 1), R (bounds between 0 and 1, strictly less than 1), and v (bounds between -inf and inf).
        Creates boolean lists to store whether each variable has been set to an ML output
        or a constant. Adds substochastic constraints on pi, strictly substochastic constraints
        on Q, and substochastic constraints on R. In pratice, strict substochastic constraints
        are implemented with a small numerical offset epsilon=1e-6. Lastly, adds the definition of v as
        v = Qv + R1.
        """
        self.variables['pi'] = self.model.addVars(
            self.n_transient,
            name="pi",
            lb=0,
            ub=1,
            vtype=GRB.CONTINUOUS
        )

        self.variables['Q'] = self.model.addVars(
            self.n_transient,
            self.n_transient,
            name="Q",
            lb=0,
            ub=1-self._epsilon,
            vtype=GRB.CONTINUOUS
        )

        self.variables['R'] = self.model.addVars(
            self.n_transient,
            self.n_targets,
            name="R",
            lb=0,
            ub=1,
            vtype=GRB.CONTINUOUS
        )

        self.variables['v'] = self.model.addVars(
            self.n_transient,
            name="v",
            lb=-GRB.INFINITY,
            ub=GRB.INFINITY,
            vtype=GRB.CONTINUOUS
        )

        self.model.update()

        self._var_is_ml_output['pi'] = [False for _ in range(self.n_transient)]
        self._var_is_const['pi'] = [False for _ in range(self.n_transient)]
        self._var_is_ml_output['Q'] = [[False for _ in range(self.n_transient)] for _ in range(self.n_transient)]
        self._var_is_const['Q'] = [[False for _ in range(self.n_transient)] for _ in range(self.n_transient)]
        self._var_is_ml_output['R'] = [[False for _ in range(self.n_targets)] for _ in range(self.n_transient)]
        self._var_is_const['R'] = [[False for _ in range(self.n_targets)] for _ in range(self.n_transient)]
        self._var_is_ml_output['v'] = [False for _ in range(self.n_transient)]
        self._var_is_const['v'] = [False for _ in range(self.n_transient)]

        self._var_dim['pi'] = (self.n_transient,)
        self._var_dim['Q'] = (self.n_transient, self.n_transient)
        self._var_dim['R'] = (self.n_transient, self.n_targets)
        self._var_dim['v'] = (self.n_transient,)

        if pi is not None:
            self.set_pi(pi)
        if Q is not None:
            self.set_Q(Q)
        if R is not None:
            self.set_R(R)

        # pi is substochastic
        self.model.addConstr(gp.quicksum(self.pi[i] for i in range(self.n_transient))
                             <= 1,
                             name="pi_prob_constr")
        # Q is strictly substochastic
        for i in range(self.n_transient):
            self.model.addConstr(gp.quicksum(self.Q[i, j] for j in range(self.n_transient))
                                 <= 1 - self._epsilon,
                                 name=f"q_prob_constr_{i}")
        # R is substochastic
        for i in range(self.n_transient):
            self.model.addConstr(gp.quicksum(self.R[i, j] for j in range(self.n_targets))
                                 <= 1,
                                 name=f"r_prob_constr_{i}")

        # v = Qv + R1
        for i in range(self.n_transient):
            R1 = gp.quicksum(self.R[i, j] for j in range(self.n_targets))
            Qv = gp.quicksum(self.Q[i, j] * self.v[j] for j in range(self.n_transient))
            self.model.addConstr(self.v[i] == Qv + R1,
                                 name=f"v_constr_{i}")

        self.model.update()

    def _check_all_vars_set(self):
        """
        Check if pi, Q, and R have been set before optimization.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If any required variables have not been set
        """
        for i in range(self._var_dim['pi'][0]):
            if not (self._var_is_ml_output['pi'][i] or self._var_is_const['pi'][i]):
                raise ValueError(f"pi[{i}] needs to be set to a constant or linked to an ML output")
        for i in range(self._var_dim['Q'][0]):
            for j in range(self._var_dim['Q'][1]):
                if not (self._var_is_ml_output['Q'][i][j] or self._var_is_const['Q'][i][j]):
                    raise ValueError(f"Q[{i}, {j}] needs to be set to a constant or linked to an ML output")
        for i in range(self._var_dim['R'][0]):
            for j in range(self._var_dim['R'][1]):
                if not (self._var_is_ml_output['R'][i][j] or self._var_is_const['R'][i][j]):
                    raise ValueError(f"R[{i}, {j}] needs to be set to a constant or linked to an ML output")

    def _set_objective(self):
        """
        Sets the objective function to pi * v.

        Returns
        -------
        None
        """
        self.model.setObjective(gp.quicksum(self.pi[i] * self.v[i] for i in range(self.n_transient)))

    def _set_feasible_constraints(self, lb : float | int,
                                 ub : float | int):
        """
        Sets the feasible constraints on the objective. When looking for a feasible solution,
        this simply adds the constraints: lb <= pi * v <= ub.

        Parameters
        ----------
        lb : float | int
            The lower bound
        ub : float | int
            The upper bound

        Returns
        -------
        None
        """
        self.model.addConstr(gp.quicksum(self.pi[i] * self.v[i] for i in range(self.n_transient)) >= lb, name="temp_feasible_lb")
        self.model.addConstr(gp.quicksum(self.pi[i] * self.v[i] for i in range(self.n_transient)) <= ub, name="temp_feasible_ub")

    def get_values(self):
        """
        Returns the values of pi, Q, R, v, features, and ml_outputs as a dictionary.

        Returns
        -------
        dict
            A dictionary with keys:
            - `pi`: the values of pi
            - `Q`: the values of Q
            - `R`: the values of R
            - `v`: the values of v
            - `features`: the values of the features
            - `ml_outputs`: the values of the ml_outputs
        """
        return {
            "pi": [self.pi[i].x for i in range(self.n_transient)],
            "Q": [[self.Q[i, j].x for j in range(self.n_transient)] for i in range(self.n_transient)],
            "R": [[self.R[i, j].x for j in range(self.n_targets)] for i in range(self.n_transient)],
            "v": [self.v[i].x for i in range(self.n_transient)],
            "features": [self.features[i].x for i in range(self.n_features)],
            "ml_outputs": [[self.ml_outputs[i][j].x for j in range(len(self.ml_outputs[i]))] for i in range(len(self.ml_outputs))]
        }

    def _compute_affine_bounds(self, ml_bounds : List[List[Tuple[float, float]]]):
        """
        Computes the affine bounds for the parameters that are linked to ML outputs.

        Parameters
        ----------
        ml_bounds : List[List[Tuple[float, float]]]
            The bounds for the ML outputs

        Returns
        -------
        dict
            A dictionary with keys: 'pi', 'Q', 'R'
        """
        # Convert ml_bounds to dictionary for easier lookup
        bounds_dict = {}
        for i, model_bounds in enumerate(ml_bounds):
            for j, (lb, ub) in enumerate(model_bounds):
                bounds_dict[f"ml_outputs_{i}[{j}]"] = (lb, ub)

        # Initialize bounds dictionary
        affine_bounds = {
            'pi': [],
            'Q': [],
            'R': []
        }

        # Compute bounds for each parameter that is an ML output
        for i in range(self._var_dim['pi'][0]):
            if self._var_is_ml_output['pi'][i]:
                expr = self._get_ml_expr(self.pi[i])
                affine_bounds['pi'].append((i, self._compute_expr_bounds(expr, bounds_dict)))

        for i in range(self._var_dim['Q'][0]):
            for j in range(self._var_dim['Q'][1]):
                if self._var_is_ml_output['Q'][i][j]:
                    expr = self._get_ml_expr(self.Q[i, j])
                    affine_bounds['Q'].append((i, j, self._compute_expr_bounds(expr, bounds_dict)))

        for i in range(self._var_dim['R'][0]):
            for j in range(self._var_dim['R'][1]):
                if self._var_is_ml_output['R'][i][j]:
                    expr = self._get_ml_expr(self.R[i, j])
                    affine_bounds['R'].append((i, j, self._compute_expr_bounds(expr, bounds_dict)))

        return affine_bounds

    def _set_affine_bounds(self, affine_bounds : dict):
        """
        Sets the bounds for the parameters that are linked to ML outputs.

        Parameters
        ----------
        affine_bounds : dict
            The bounds for the ML outputs

        Returns
        -------
        None
        """
        # Set bounds only for parameters that are ML outputs
        for i, (lb, ub) in affine_bounds['pi']:
            self._update_lb(self.pi[i], lb)
            self._update_ub(self.pi[i], ub)

        for i, j, (lb, ub) in affine_bounds['Q']:
            self._update_lb(self.Q[i, j], lb)
            self._update_ub(self.Q[i, j], ub)

        for i, j, (lb, ub) in affine_bounds['R']:
            self._update_lb(self.R[i, j], lb)
            self._update_ub(self.R[i, j], ub)

    def _compute_initial_v_bounds(self):
        """
        Computes the initial bounds for v using the bounds on R.
        Uses the lemma from the paper: for each $v_i$, $v_i \\in [v^min_i, v^max_i]$ where
        $v^min_i = 0$, and
        $v^max_i = 1/\\epsilon * \\max_j \\sum_{k=1}^{|S|} R_{jk}^max$,
        where $\\epsilon$ is the small numerical offset for the strict substochasticity

        Returns
        -------
        List[Tuple[float, float]]
            The initial bounds for v
        """
        # Find maximum row sum of R's upper bounds
        max_row_sum = 0
        for j in range(self.n_transient):
            row_sum = sum(self.R[j, k].UB for k in range(self.n_targets))
            max_row_sum = max(max_row_sum, row_sum)

        # Compute upper bound using epsilon
        v_max = max_row_sum / self._epsilon

        # Return bounds for each transient state: [0, v_max]
        return [(0, v_max) for _ in range(self.n_transient)]

    def _set_initial_v_bounds(self, initial_v_bounds : List[Tuple[float, float]]):
        """
        Sets the initial bounds for v.

        Parameters
        ----------
        initial_v_bounds : List[Tuple[float, float]]
            The initial bounds for v

        Returns
        -------
        None
        """
        for i in range(self._var_dim['v'][0]):
            lb, ub = initial_v_bounds[i]
            self._update_lb(self.v[i], lb)
            self._update_ub(self.v[i], ub)

    def _compute_tightened_v_bounds(self):
        """
        Computes the tightened bounds for v using the bounds for Q and R and initial bounds for v.
        This uses Gauss-Seidel to solve the linear system: $vM = R1$, where $M = I - Q$.

        Returns
        -------
        List[Tuple[float, float]]
            The tightened bounds for v
        """
        # Create interval matrices for Q and R
        Q_intervals = np.zeros((self._var_dim['Q'][0], self._var_dim['Q'][1], 2))
        R_intervals = np.zeros((self._var_dim['R'][0], self._var_dim['R'][1], 2))

        # Fill intervals from current bounds
        for i in range(self._var_dim['Q'][0]):
            for j in range(self._var_dim['Q'][1]):
                Q_intervals[i,j] = [self.Q[i,j].LB, self.Q[i,j].UB]
        for i in range(self._var_dim['R'][0]):
            for j in range(self._var_dim['R'][1]):
                R_intervals[i,j] = [self.R[i,j].LB, self.R[i,j].UB]

        R1_intervals = R_intervals.sum(axis=1) # R1 is basically row sums

        # Create linear system: Mv = R1, where M = I - Q
        I = ima._identity_interval(self._var_dim['v'][0])
        M = ima.subtract(I, Q_intervals)

        # Initial guess using current v bounds
        v0 = np.array([[self.v[i].LB, self.v[i].UB] for i in range(self._var_dim['v'][0])])

        # Solve using Gauss-Seidel
        v_intervals = ima.interval_gauss_seidel(
            M, R1_intervals, v0,
            max_iter=100,
            tol=1e-6
        )

        # Convert to list of tuples
        return [(v_intervals[i][0], v_intervals[i][1]) for i in range(self._var_dim['v'][0])]

    def _set_tightened_v_bounds(self, tightened_v_bounds : List[Tuple[float, float]]):
        """
        Sets the tightened bounds for v.

        Parameters
        ----------
        tightened_v_bounds : List[Tuple[float, float]]
            The tightened bounds for v

        Returns
        -------
        None
        """
        for i in range(self._var_dim['v'][0]):
            lb, ub = tightened_v_bounds[i]
            self._update_lb(self.v[i], lb)
            self._update_ub(self.v[i], ub)

    # CLEAN PARAMETER ACCESSORS

    @property
    def pi(self):
        """
        Accessor for the pi parameter.

        Returns
        -------
        gurobipy.tupledict
            The pi parameter
        """
        return self.variables['pi']

    @property
    def Q(self):
        """
        Accessor for the Q parameter.

        Returns
        -------
        gurobipy.tupledict
            The Q parameter
        """
        return self.variables['Q']

    @property
    def R(self):
        """
        Accessor for the R parameter.

        Returns
        -------
        gurobipy.tupledict
            The R parameter
        """
        return self.variables['R']

    @property
    def v(self):
        """
        Accessor for the v parameter.

        Returns
        -------
        gurobipy.tupledict
            The v parameter
        """
        return self.variables['v']

    # HELPER FUNCTIONS TO SET PARAMETERS

    def set_pi(self, values: List[float | int | gp.Var | gp.LinExpr]):
        """
        Helper function to set the entire pi vector.

        Parameters
        ----------
        values : List[float | int | gp.Var | gp.LinExpr]
            The values to set the pi parameter to

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the values have incorrect length, are not in [0, 1], don't sum to <=1, or pi is already set

        Notes
        -----
        This checks for validity of the values, checks if pi is already set, and sets the values.
        """
        self._validate_vector_length(values, "pi")
        if self._seq_is_all_const(values):
            self._validate_vector_bounds(values, "pi", (0, 1))
            self._validate_stochastic_vector(values, "pi", sub=True)
        assert not self._check_any_var_set("pi"),\
              "pi is already set"

        for i in range(self._var_dim["pi"][0]):
            if isinstance(values[i], (float, int)):
                self.set_to_const(self.pi[i], values[i])
            elif isinstance(values[i], (gp.Var, gp.LinExpr)):
                self.set_to_ml_output(self.pi[i], values[i])
            else:
                raise ValueError(f"pi[{i}] must be a float, int, or a Gurobi expression")

    def set_Q(self, values: List[List[float | int | gp.Var | gp.LinExpr]]):
        """
        Helper function to set the Q matrix.

        Parameters
        ----------
        values : List[List[float | int | gp.Var | gp.LinExpr]]
            The values to set the Q matrix to

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the values have incorrect length, are not in [0, 1), rows don't sum to strictly less than 1, or Q is already set

        Notes
        -----
        This checks for validity of the values, checks if Q is already set, and sets the values.
        """
        self._validate_matrix_shape(values, "Q")
        if self._seq_is_all_const(values):
            self._validate_matrix_bounds(values, "Q", (0, 1))
            self._validate_stochastic_matrix(values, "Q", strictsub=True) # Q is strictly substochastic
        for i in range(len(values)):
            if self._seq_is_all_const(values[i]):
                self._validate_stochastic_vector(values[i], f"Q[{i}]", strictsub=True) # Q is strictly substochastic
        assert not self._check_any_var_set("Q"),\
              "Q is already set"

        for i in range(self._var_dim["Q"][0]):
            for j in range(self._var_dim["Q"][1]):
                if isinstance(values[i][j], (float, int)):
                    self.set_to_const(self.Q[i, j], values[i][j])
                elif isinstance(values[i][j], (gp.Var, gp.LinExpr)):
                    self.set_to_ml_output(self.Q[i, j], values[i][j])
                else:
                    raise ValueError(f"Q[{i}, {j}] must be a float, int, or a Gurobi expression")

    def set_R(self, values: List[List[float | int | gp.Var | gp.LinExpr]]):
        """
        Helper function to set the R matrix.

        Parameters
        ----------
        values : List[List[float | int | gp.Var | gp.LinExpr]]
            The values to set the R matrix to

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the values have incorrect length, are not in [0, 1], rows don't sum to <=1, or R is already set

        Notes
        -----
        This checks for validity of the values, checks if R is already set, and sets the values.
        """
        self._validate_matrix_shape(values, "R")
        if self._seq_is_all_const(values):
            self._validate_matrix_bounds(values, "R", (0, 1))
            self._validate_stochastic_matrix(values, "R", sub=True)
        for i in range(len(values)):
            if self._seq_is_all_const(values[i]):
                self._validate_stochastic_vector(values[i], f"R[{i}]", sub=True)
        assert not self._check_any_var_set("R"),\
              "R is already set"

        for i in range(self._var_dim["R"][0]):
            for j in range(self._var_dim["R"][1]):
                if isinstance(values[i][j], (float, int)):
                    self.set_to_const(self.R[i, j], values[i][j])
                elif isinstance(values[i][j], (gp.Var, gp.LinExpr)):
                    self.set_to_ml_output(self.R[i, j], values[i][j])
                else:
                    raise ValueError(f"R[{i}, {j}] must be a float, int, or a Gurobi expression")

class MarkovHitting(AbstractMarkov):
    """
    For the hitting time problem.
    """
    def __init__(self, n_states: int,
                 n_features: int,
                 n_transient: int,
                 pi: Optional[List | np.ndarray] = None,
                 Q: Optional[List[List] | np.ndarray] = None):
        """
        Initialize a Markov chain for the hitting time problem.

        Parameters
        ----------
        n_states : int
            The number of states
        n_features : int
            The number of features
        n_transient : int
            The number of transient states ($|T|$ in the paper)
        pi : Optional[List | np.ndarray]
            The initial state distribution over transient states
        Q : Optional[List[List] | np.ndarray]
            The transition matrix between transient states

        Raises
        -------
        ValueError
            If the number of transient states is greater than or equal to the number of states

        Notes
        -----
        Calls the constructor of the parent class AbstractMarkov, with additional attributes.
        Then, initializes the defining parameters and constraints for the hitting time problem.
        """

        if n_transient >= n_states:
            raise ValueError("Number of transient states must be less than number of states")

        super().__init__(n_states, n_features)
        self.n_transient = n_transient
        self._initialize_parameters_and_constraints(pi, Q)

    # FUNCTIONS THAT MUST BE IMPLEMENTED IN SUBCLASSES WHICH UNIQUELY DEFINE THE PROBLEM

    def _initialize_parameters_and_constraints(self, pi : Optional[List | np.ndarray] = None,
                                               Q : Optional[List[List] | np.ndarray] = None):
        """
        Initializes the parameters and constraints for the reachability problem.

        Parameters
        ----------
        pi : Optional[List | np.ndarray]
            The initial state distribution over transient states
        Q : Optional[List[List] | np.ndarray]
            The transition matrix between transient states

        Returns
        -------
        None

        Notes
        -----
        Adds variables for: pi (bounds between 0 and 1), Q (bounds between 0 and 1, strictly less
        than 1), and v (bounds between -inf and inf). Creates boolean lists to store whether each
        variable has been set to an ML output or a constant. Adds substochastic constraints on pi,
        and strictly substochastic constraints on Q. In pratice, strict substochastic constraints
        are implemented with a small numerical offset epsilon=1e-6. Lastly, adds the definition
        of v as v = Qv + 1.
        """
        self.variables['pi'] = self.model.addVars(
            self.n_transient,
            name="pi",
            lb=0,
            ub=1,
            vtype=GRB.CONTINUOUS
        )

        self.variables['Q'] = self.model.addVars(
            self.n_transient,
            self.n_transient,
            name="Q",
            lb=0,
            ub=1-self._epsilon,
            vtype=GRB.CONTINUOUS
        )

        self.variables['v'] = self.model.addVars(
            self.n_transient,
            name="v",
            lb=-GRB.INFINITY,
            ub=GRB.INFINITY,
            vtype=GRB.CONTINUOUS
        )

        self.model.update()

        self._var_is_ml_output['pi'] = [False for _ in range(self.n_transient)]
        self._var_is_const['pi'] = [False for _ in range(self.n_transient)]
        self._var_is_ml_output['Q'] = [[False for _ in range(self.n_transient)] for _ in range(self.n_transient)]
        self._var_is_const['Q'] = [[False for _ in range(self.n_transient)] for _ in range(self.n_transient)]
        self._var_is_ml_output['v'] = [False for _ in range(self.n_transient)]
        self._var_is_const['v'] = [False for _ in range(self.n_transient)]

        self._var_dim['pi'] = (self.n_transient,)
        self._var_dim['Q'] = (self.n_transient, self.n_transient)
        self._var_dim['v'] = (self.n_transient,)

        if pi is not None:
            self.set_pi(pi)
        if Q is not None:
            self.set_Q(Q)

        # pi is substochastic
        self.model.addConstr(gp.quicksum(self.pi[i] for i in range(self.n_transient))
                             <= 1,
                             name="pi_prob_constr")
        # Q is strictly substochastic
        for i in range(self.n_transient):
            self.model.addConstr(gp.quicksum(self.Q[i, j] for j in range(self.n_transient))
                                 <= 1 - self._epsilon,
                                 name=f"q_prob_constr_{i}")

        # v = Qv + 1
        for i in range(self.n_transient):
            Qv = gp.quicksum(self.Q[i, j] * self.v[j] for j in range(self.n_transient))
            self.model.addConstr(self.v[i] == Qv + 1,
                                 name=f"v_constr_{i}")

        self.model.update()

    def _check_all_vars_set(self):
        """
        Check if pi and Q have been set before optimization.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If any required variables have not been set
        """
        for i in range(self._var_dim['pi'][0]):
            if not (self._var_is_ml_output['pi'][i] or self._var_is_const['pi'][i]):
                raise ValueError(f"pi[{i}] needs to be set to a constant or linked to an ML output")
        for i in range(self._var_dim['Q'][0]):
            for j in range(self._var_dim['Q'][1]):
                if not (self._var_is_ml_output['Q'][i][j] or self._var_is_const['Q'][i][j]):
                    raise ValueError(f"Q[{i}, {j}] needs to be set to a constant or linked to an ML output")

    def _set_objective(self):
        """
        Sets the objective function to pi * v.

        Returns
        -------
        None
        """
        self.model.setObjective(gp.quicksum(self.pi[i] * self.v[i] for i in range(self.n_transient)))

    def _set_feasible_constraints(self, lb : float | int,
                                 ub : float | int):
        """
        Sets the feasible constraints on the objective. When looking for a feasible solution,
        this simply adds the constraints: lb <= pi * v <= ub.

        Parameters
        ----------
        lb : float | int
            The lower bound
        ub : float | int
            The upper bound

        Returns
        -------
        None
        """
        self.model.addConstr(gp.quicksum(self.pi[i] * self.v[i] for i in range(self.n_transient)) >= lb, name="temp_feasible_lb")
        self.model.addConstr(gp.quicksum(self.pi[i] * self.v[i] for i in range(self.n_transient)) <= ub, name="temp_feasible_ub")

    def get_values(self):
        """
        Returns the values of pi, Q, v, features, and ml_outputs as a dictionary.

        Returns
        -------
        dict
            A dictionary with keys:
            - `pi`: the values of pi
            - `Q`: the values of Q
            - `v`: the values of v
            - `features`: the values of the features
            - `ml_outputs`: the values of the ml_outputs
        """
        return {
            "pi": [self.pi[i].x for i in range(self.n_transient)],
            "Q": [[self.Q[i, j].x for j in range(self.n_transient)] for i in range(self.n_transient)],
            "v": [self.v[i].x for i in range(self.n_transient)],
            "features": [self.features[i].x for i in range(self.n_features)],
            "ml_outputs": [[self.ml_outputs[i][j].x for j in range(len(self.ml_outputs[i]))] for i in range(len(self.ml_outputs))]
        }

    def _compute_affine_bounds(self, ml_bounds : List[List[Tuple[float, float]]]):
        """
        Computes the affine bounds for the parameters that are linked to ML outputs.

        Parameters
        ----------
        ml_bounds : List[List[Tuple[float, float]]]
            The bounds for the ML outputs

        Returns
        -------
        dict
            A dictionary with keys: 'pi', and 'Q'
        """
        # Convert ml_bounds to dictionary for easier lookup
        bounds_dict = {}
        for i, model_bounds in enumerate(ml_bounds):
            for j, (lb, ub) in enumerate(model_bounds):
                bounds_dict[f"ml_outputs_{i}[{j}]"] = (lb, ub)

        # Initialize bounds dictionary
        affine_bounds = {
            'pi': [],
            'Q': []
        }

        # Compute bounds for each parameter that is an ML output
        for i in range(self._var_dim['pi'][0]):
            if self._var_is_ml_output['pi'][i]:
                expr = self._get_ml_expr(self.pi[i])
                affine_bounds['pi'].append((i, self._compute_expr_bounds(expr, bounds_dict)))

        for i in range(self._var_dim['Q'][0]):
            for j in range(self._var_dim['Q'][1]):
                if self._var_is_ml_output['Q'][i][j]:
                    expr = self._get_ml_expr(self.Q[i, j])
                    affine_bounds['Q'].append((i, j, self._compute_expr_bounds(expr, bounds_dict)))

        return affine_bounds

    def _set_affine_bounds(self, affine_bounds : dict):
        """
        Sets the bounds for the parameters that are linked to ML outputs.

        Parameters
        ----------
        affine_bounds : dict
            The bounds for the ML outputs

        Returns
        -------
        None
        """
        # Set bounds only for parameters that are ML outputs
        for i, (lb, ub) in affine_bounds['pi']:
            self._update_lb(self.pi[i], lb)
            self._update_ub(self.pi[i], ub)

        for i, j, (lb, ub) in affine_bounds['Q']:
            self._update_lb(self.Q[i, j], lb)
            self._update_ub(self.Q[i, j], ub)

    def _compute_initial_v_bounds(self):
        """
        Computes the initial bounds for v. Simply, for each $v_i$, $ 0 \leq v_i \leq 1 / \\epsilon$ where
        $\\epsilon$ is the small numerical offset for the strict substochasticity.

        Returns
        -------
        List[Tuple[float, float]]
            The initial bounds for v
        """
        # Compute v bounds using epsilon
        v_min = 0
        v_max = 1 / self._epsilon

        # Return bounds for each state
        return [(v_min, v_max) for _ in range(self._var_dim['v'][0])]

    def _set_initial_v_bounds(self, initial_v_bounds : List[Tuple[float, float]]):
        """
        Sets the initial bounds for v.

        Parameters
        ----------
        initial_v_bounds : List[Tuple[float, float]]
            The initial bounds for v

        Returns
        -------
        None
        """
        for i in range(self._var_dim['v'][0]):
            lb, ub = initial_v_bounds[i]
            self._update_lb(self.v[i], lb)
            self._update_ub(self.v[i], ub)

    def _compute_tightened_v_bounds(self):
        """
        Computes the tightened bounds for v using the bounds for Q and initial bounds for v.
        This uses Gauss-Seidel to solve the linear system: $vM = 1$, where $M = I - Q$.

        Returns
        -------
        List[Tuple[float, float]]
            The tightened bounds for v
        """
        # Create interval matrices for Q
        Q_intervals = np.zeros((self._var_dim['Q'][0], self._var_dim['Q'][1], 2))

        # Fill intervals from current bounds
        for i in range(self._var_dim['Q'][0]):
            for j in range(self._var_dim['Q'][1]):
                Q_intervals[i,j] = [self.Q[i,j].LB, self.Q[i,j].UB]

        # Create linear system: Mv = 1, where M = I - Q
        I = ima._identity_interval(self._var_dim['v'][0])
        M = ima.subtract(I, Q_intervals)
        ones_intervals = np.ones((self._var_dim['v'][0], 2))

        # Initial guess using current v bounds
        v0 = np.array([[self.v[i].LB, self.v[i].UB] for i in range(self._var_dim['v'][0])])

        # Solve using Gauss-Seidel
        v_intervals = ima.interval_gauss_seidel(
            M, ones_intervals, v0,
            max_iter=100,
            tol=1e-6
        )

        # Convert to list of tuples
        return [(v_intervals[i][0], v_intervals[i][1]) for i in range(self._var_dim['v'][0])]

    def _set_tightened_v_bounds(self, tightened_v_bounds : List[Tuple[float, float]]):
        """
        Sets the tightened bounds for v.

        Parameters
        ----------
        tightened_v_bounds : List[Tuple[float, float]]
            The tightened bounds for v

        Returns
        -------
        None
        """
        for i in range(self._var_dim['v'][0]):
            lb, ub = tightened_v_bounds[i]
            self._update_lb(self.v[i], lb)
            self._update_ub(self.v[i], ub)

    # CLEAN PARAMETER ACCESSORS

    @property
    def pi(self):
        """
        Accessor for the pi parameter.

        Returns
        -------
        gurobipy.tupledict
            The pi parameter
        """
        return self.variables['pi']

    @property
    def Q(self):
        """
        Accessor for the Q parameter.

        Returns
        -------
        gurobipy.tupledict
            The Q parameter
        """
        return self.variables['Q']

    @property
    def v(self):
        """
        Accessor for the v parameter.

        Returns
        -------
        gurobipy.tupledict
            The v parameter
        """
        return self.variables['v']

    # HELPER FUNCTIONS TO SET PARAMETERS

    def set_pi(self, values: List[float | int | gp.Var | gp.LinExpr]):
        """
        Helper function to set the entire pi vector.

        Parameters
        ----------
        values : List[float | int | gp.Var | gp.LinExpr]
            The values to set the pi parameter to

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the values have incorrect length, are not in [0, 1], don't sum to <=1, or pi is already set

        Notes
        -----
        This checks for validity of the values, checks if pi is already set, and sets the values.
        """
        self._validate_vector_length(values, "pi")
        if self._seq_is_all_const(values):
            self._validate_vector_bounds(values, "pi", (0, 1))
            self._validate_stochastic_vector(values, "pi", sub=True)
        assert not self._check_any_var_set("pi"),\
              "pi is already set"

        for i in range(self._var_dim["pi"][0]):
            if isinstance(values[i], (float, int)):
                self.set_to_const(self.pi[i], values[i])
            elif isinstance(values[i], (gp.Var, gp.LinExpr)):
                self.set_to_ml_output(self.pi[i], values[i])
            else:
                raise ValueError(f"pi[{i}] must be a float, int, or a Gurobi expression")

    def set_Q(self, values: List[List[float | int | gp.Var | gp.LinExpr]]):
        """
        Helper function to set the Q matrix.

        Parameters
        ----------
        values : List[List[float | int | gp.Var | gp.LinExpr]]
            The values to set the Q matrix to

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the values have incorrect length, are not in [0, 1), rows don't sum to strictly less than 1, or Q is already set

        Notes
        -----
        This checks for validity of the values, checks if Q is already set, and sets the values.
        """
        self._validate_matrix_shape(values, "Q")
        if self._seq_is_all_const(values):
            self._validate_matrix_bounds(values, "Q", (0, 1))
            self._validate_stochastic_matrix(values, "Q", strictsub=True) # Q is strictly substochastic
        for i in range(len(values)):
            if self._seq_is_all_const(values[i]):
                self._validate_stochastic_vector(values[i], f"Q[{i}]", strictsub=True) # Q is strictly substochastic
        assert not self._check_any_var_set("Q"),\
              "Q is already set"

        for i in range(self._var_dim["Q"][0]):
            for j in range(self._var_dim["Q"][1]):
                if isinstance(values[i][j], (float, int)):
                    self.set_to_const(self.Q[i, j], values[i][j])
                elif isinstance(values[i][j], (gp.Var, gp.LinExpr)):
                    self.set_to_ml_output(self.Q[i, j], values[i][j])
                else:
                    raise ValueError(f"Q[{i}, {j}] must be a float, int, or a Gurobi expression")
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from gurobi_ml.sklearn import (add_mlp_regressor_constr,
                               add_decision_tree_regressor_constr,
                               add_random_forest_regressor_constr)
from gurobi_ml.torch import add_sequential_constr
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from gurobipy import nlfunc
from markovml.utils.models_ext import DecisionRules, SequentialClassifier
from torch import nn
from torch.nn import Sequential

def add_mlp_classifier_constr(model: gp.Model, clf: MLPClassifier, input_vars, output_vars):
    """
    Encodes an MLPClassifier in a Gurobi model by manually implementing softmax activation.
    Uses Gurobi's nonlinear functions to implement softmax.

    Parameters
    ----------
    model : gp.Model
        Model to add constraints to
    clf : MLPClassifier
        Classifier to add constraints for
    input_vars : List[gp.Var]
        Variables corresponding to features
    output_vars : List[gp.Var]
        Variables to store probabilities (e.g., P[0,:])

    Returns
    -------
    None
    """
    n = len(output_vars)

    assert n == clf.n_outputs_, "Number of output variables must match number of classes"

    # Get logits from network
    logits = model.addVars(n, lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"MLPClassifier_Logits")

    # Temporarily change activation to get raw logits
    original_activation = clf.out_activation_
    clf.out_activation_ = "identity"
    pred_constr = add_mlp_regressor_constr(model, clf, input_vars, logits)
    clf.out_activation_ = original_activation

    # Compute softmax
    # softmax: P = exp(logits)/sum(exp(logits))
    sum_exp = gp.quicksum(nlfunc.exp(logits[i]) for i in range(n))
    for i in range(n):
        model.addConstr(
            output_vars[i] == nlfunc.exp(logits[i]) / sum_exp,
            name=f"MLPClassifier_Softmax_{i}"
        )

    return

def add_decision_tree_classifier_constr(model: gp.Model, clf: DecisionTreeClassifier, input_vars, output_vars):
    """
    Encodes a DecisionTreeClassifier in a Gurobi model.

    Parameters
    ----------
    model : gp.Model
        Model to add constraints to
    clf : DecisionTreeClassifier
        Classifier to add constraints for
    input_vars : List[gp.Var]
        Variables corresponding to features
    output_vars : List[gp.Var]
        Variables to store outputs (e.g., P[0,:])

    Returns
    -------
    None
    """
    add_decision_tree_regressor_constr(model, clf, input_vars, output_vars)

def add_random_forest_classifier_constr(model: gp.Model, clf: RandomForestClassifier, input_vars, output_vars):
    """
    Encodes a RandomForestClassifier in a Gurobi model.

    Parameters
    ----------
    model : gp.Model
        Model to add constraints to
    clf : RandomForestClassifier
        Classifier to add constraints for
    input_vars : List[gp.Var]
        Variables corresponding to features
    output_vars : List[gp.Var]
        Variables to store outputs (e.g., P[0,:])

    Returns
    -------
    None
    """
    add_random_forest_regressor_constr(model, clf, input_vars, output_vars)

def add_decision_rules_constr(model: gp.Model, rules: DecisionRules, input_vars, output_vars):
    """
    Add constraints to encode decision rules in a Gurobi model.

    Parameters
    ----------
    model : gp.Model
        Gurobi model to add constraints to
    rules : DecisionRules
        Fitted DecisionRules object
    input_vars : List[gp.Var]
        Input variables corresponding to features
    output_vars : List[gp.Var]
        Output variables (note that only one output variable is supported for now)

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If DecisionRules is not fitted or if there is more than one output variable

    Notes
    -----
    Uses a series of logical implications to encode the rules. Note that the encoding makes
    it so that the first rule that an input satisfies is the one that is used.
    """
    if rules._compiled_rules is None:
        raise ValueError("DecisionRules must be fitted before adding constraints")
    if len(output_vars) != 1:
        raise ValueError("DecisionRules must have only one output variable")

    # numerical tolerance for strict inequalities
    epsilon = 1e-6
    output_var = output_vars[0]
    if not isinstance(input_vars, list):
        input_vars = list(input_vars.values()) # if its dict or gurobi tupledict

    # binary variable for each rule being true
    rule_binaries = model.addVars(len(rules._compiled_rules), name="rule")

    for i, rule in enumerate(rules._compiled_rules):
        if rule['type'] == 'if':
            clauses = rule['clauses']
            # binary variable for each clause being true
            clause_binaries = model.addVars(len(clauses), name=f"rule_{i}_clause")
            for j, clause in enumerate(clauses):
                # need to encode biimplication: clause is true if and only if inequality is true
                # clause true -> inequality true
                # clause false -> inequality false
                feat, op, val = clause
                feat_idx = rules.features.index(feat)

                if op == '>':
                    model.addConstr((clause_binaries[j] == 1) >> (input_vars[feat_idx] >= val + epsilon))
                    model.addConstr((clause_binaries[j] == 0) >> (input_vars[feat_idx] <= val))
                elif op == '>=':
                    model.addConstr((clause_binaries[j] == 1) >> (input_vars[feat_idx] >= val))
                    model.addConstr((clause_binaries[j] == 0) >> (input_vars[feat_idx] <= val - epsilon))
                elif op == '<':
                    model.addConstr((clause_binaries[j] == 1) >> (input_vars[feat_idx] <= val - epsilon))
                    model.addConstr((clause_binaries[j] == 0) >> (input_vars[feat_idx] >= val))
                elif op == '<=':
                    model.addConstr((clause_binaries[j] == 1) >> (input_vars[feat_idx] <= val))
                    model.addConstr((clause_binaries[j] == 0) >> (input_vars[feat_idx] >= val + epsilon))

            # rule is true if and only if all clauses are true and all previous rules are false
            # using rules of logical implication, we write the following:
                # 1. rule true -> all clauses true
                # 2. rule true -> all previous rules false
                # 3. rule false -> not (all clauses true) or not (all previous rules false)
                #   rule false -> clause_1 false or clause_2 false ... or prev_1 true or prev_2 true ...
                #   handle this with sums of the binary variables, setting >= 1

            # 1
            for j in range(len(clauses)):
                model.addConstr((rule_binaries[i] == 1) >> (clause_binaries[j] == 1))

            # 2
            for j in range(i):
                model.addConstr((rule_binaries[i] == 1) >> (rule_binaries[j] == 0))

            # 3
            model.addConstr((rule_binaries[i] == 0) >> (
                gp.quicksum(1 - clause_binaries[j] for j in range(len(clauses))) +
                gp.quicksum(rule_binaries[j] for j in range(i)) >= 1
            ))

            # if rule is true then fix output according to rule
            model.addConstr((rule_binaries[i] == 1) >> (output_var == rule['output']))

        elif rule['type'] == 'else':
            # fix output according to else rule
            model.addConstr((rule_binaries[i] == 1) >> (output_var == rule['output']))

    # only one rule can be true
    model.addConstr(rule_binaries.sum() == 1)

    return

def add_sequential_classifier_constr(model: gp.Model, clf: SequentialClassifier, input_vars, output_vars):
    """
    Encodes a SequentialClassifier in a Gurobi model by manually implementing softmax activation.
    SequentialClassifier is a wrapper around a PyTorch Sequential model that adds a final softmax layer.
    Uses Gurobi's nonlinear functions to implement softmax.

    Parameters
    ----------
    model : gp.Model
        Model to add constraints to
    clf : SequentialClassifier
        Classifier to add constraints for
    input_vars : List[gp.Var]
        Variables corresponding to features
    output_vars : List[gp.Var]
        Variables to store probabilities (e.g., P[0,:])

    Returns
    -------
    None
    """
    # Get all layers except the final Softmax
    clf_before_softmax = Sequential(*list(clf.children())[:-1])

    # Create intermediate variables for the logits (before softmax)
    n_classes = _get_sequential_n_outputs(clf_before_softmax)
    logits = model.addVars(n_classes, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="sequential_classifier_logits")

    # Add constraints for all layers before softmax
    add_sequential_constr(model, clf_before_softmax, input_vars, logits)

    # Add softmax constraints
    sum_exp = gp.quicksum(nlfunc.exp(logits[i]) for i in range(n_classes))
    for i in range(n_classes):
        model.addConstr(
            output_vars[i] == nlfunc.exp(logits[i]) / sum_exp,
            name=f"sequential_classifier_softmax_{i}"
        )

    return

def _get_sequential_n_outputs(model: Sequential | SequentialClassifier):
    """Get number of outputs from last Linear layer, ignoring any activation layers after it."""
    for layer in reversed(list(model.children())):
        if isinstance(layer, nn.Linear):
            return layer.out_features
    raise ValueError("No Linear layer found in model")
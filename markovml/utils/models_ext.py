import numpy as np
from torch import nn
from typing import List

class DecisionRules:
    """
    Decision rules is a "model" that is a set of rules.
    It implements a series of if-then rules, and an else rule.
    It can be used to encode simple decision rules as are found
    in the the literature. E.g., "if age > 65 then mortality risk
    is 10%".

    Example:
    ```python
    features = ['age', 'income']
    model = DecisionRules(features)
    rules = [
        "if age > 20 then 2.5",
        "if income >= 50000 and age < 30 then -1.0",
        "else 0.0"
    ]
    model.fit(rules=rules)
    ```

    You can then use the model to make predictions:
    ```python
    X = np.array([[25, 60000], [19, 30000]])
    predictions = model.predict(X)
    ```

    The first row is 2.5, the second row is 0.0.
    """

    def __init__(self, features : list[str]):
        """
        Initialize the DecisionRules model.

        Parameters
        ----------
        features : list of str
            The features that the model will use. E.g., ['age', 'income'].

        Returns
        -------
        None
        """
        self.features = features
        self._compiled_rules = None

    def _parse_clause(self, clause):
        """Parse clause like 'age > 20' or 'age>20' into (feature, operator, value)"""
        clause = clause.strip()
        # Check for each possible operator
        for op in ['>=', '<=', '>', '<', '==']:
            if op in clause:
                # Split on operator
                parts = clause.split(op)
                return (parts[0].strip(), op, float(parts[1]))

        raise ValueError(f"No valid operator found in clause: {clause}")

    def fit(self, rules : list[str]) -> 'DecisionRules':
        """
        Fits the DecisionRules model to the provided rules.

        The rules are specified in natural language format, with each rule being
        a string like "if age > 20 then 2.5".

        Parameters
        ----------
        rules : list of str
            The rules to compile. Each rule should be in the format
            "if <condition> then <value>" or "else <value>".

        Returns
        -------
        DecisionRules
            The fitted model instance.

        Raises
        ------
        ValueError
            If any of the following conditions are met:
            * Multiple else clauses are found
            * The else clause is not the last rule
            * No rules are provided
            * A rule contains an operator other than >, >=, <, <=, ==
        """
        compiled = []
        found_else = False

        for i, rule in enumerate(rules):
            if rule.startswith('else'):
                if found_else:
                    raise ValueError("Multiple else clauses found")
                if i != len(rules) - 1:
                    raise ValueError("Else clause must be the last rule")
                found_else = True
                compiled.append({
                    'type': 'else',
                    'output': float(rule.split()[-1])
                })
                continue

            condition, output = rule.split(' then ')
            clauses = condition[3:].split(' and ')  # remove 'if ' prefix
            parsed_clauses = [self._parse_clause(clause) for clause in clauses]

            compiled.append({
                'type': 'if',
                'clauses': parsed_clauses,  # list of (feature, operator, value) tuples
                'output': float(output)
            })

        if not found_else:
            raise ValueError("No else clause found")

        self._compiled_rules = compiled
        return self

    def predict(self, X : np.ndarray | List[dict]) -> np.ndarray:
        """
        Make predictions using the compiled rules.

        Parameters
        ----------
        X : np.ndarray or List[dict]
            The input data.
            - If a numpy array, it is assumed to be a 2D array where each row is
              an entry, in the order of the features passed at initialization.
            - If a list of dictionaries, each key is a feature name, and the
              value is the feature value.

        Returns
        -------
        np.ndarray
            The predictions for each input row.
        """
        if self._compiled_rules is None:
            raise ValueError("Call fit(rules) before predict")

        # Convert X to list of dictionaries
        if isinstance(X, np.ndarray):
            X_dict = [{self.features[i]: row[i] for i in range(len(self.features))}
                    for row in X]
        elif hasattr(X, 'to_dict'):
            X_dict = X.to_dict('records')
        else:
            X_dict = X

        predictions = []
        for x_row in X_dict:
            for rule in self._compiled_rules:
                if rule['type'] == 'else':
                    predictions.append(rule['output'])
                    break

                # Check if all clauses are satisfied
                all_satisfied = all(self._evaluate_clause(feat, op, val, x_row[feat])\
                                     for feat, op, val in rule['clauses'])

                if all_satisfied:
                    predictions.append(rule['output'])
                    break

        return np.array(predictions)

    def _evaluate_clause(self, feat, op, val, x_val):
        """Safely evaluate a single clause against a feature value."""
        if op == '>':
            return x_val > val
        elif op == '>=':
            return x_val >= val
        elif op == '<':
            return x_val < val
        elif op == '<=':
            return x_val <= val
        elif op == '==':
            return x_val == val
        else:
            raise ValueError(f"Unknown operator: {op}")

class SequentialClassifier(nn.Sequential):
    """Wrapper class that adds a Softmax layer to a Sequential model"""
    def __init__(self, sequential: nn.Sequential):
        # Initialize parent class with all layers plus Softmax
        super().__init__(
            *list(sequential.children()),
            nn.Softmax(dim=1)
        )
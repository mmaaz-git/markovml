# TESTING THE INTERVAL MATRIX ANALYSIS FUNCTIONS

import os, sys; sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from markovml.utils.ima import *
from markovml.utils.ima import _identity_interval

if __name__ == "__main__":

    # Example usage with interval matrice
    lambda_val = 0.97  # discount factor

    P = np.array([
        [[0.5, 0.6], [0.2, 0.5]],
        [[0.1, 0.3], [0.5, 0.6]]
    ])

    r = np.array([
        [0, 100],
        [50, 200]
    ])

    P_max = P[..., 1]
    print(f"spectral radius of P_max: {np.max(np.abs(np.linalg.eigvals(P_max)))}")
    print(f"spectral radius less than 1/lambda: {np.max(np.abs(np.linalg.eigvals(P_max))) < 1/lambda_val}")

    # Create I - lambda*P for some lambda
    n = P.shape[0]
    I = _identity_interval(n)
    # Compute I - lambda*P
    M = subtract(I, lambda_val * P)

    # Solve
    v_min = np.min(r[..., 0]) / (1 - lambda_val)  # Min of lower bounds / (1-λ)
    v_max = np.max(r[..., 1]) / (1 - lambda_val)  # Max of upper bounds / (1-λ)
    v0 = np.array([[v_min, v_max] for _ in range(n)])

    print("P:\n", P)
    print("I:\n", I)
    print("\nM = I - λP:\n", M)
    print("\nInitial v0:\n", v0)

    print("\nv from gauss-seidel: \n", interval_gauss_seidel(M, r, v0))

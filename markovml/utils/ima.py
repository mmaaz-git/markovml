# INTERVAL MATRIX ANALYSIS

import numpy as np

def negate(A):
    """Negate an interval matrix A: -[a,b] = [-b,-a]"""
    return np.stack([
        -A[..., 1],  # new min is negative of old max
        -A[..., 0]   # new max is negative of old min
    ], axis=-1)

def add(A, B):
    """Add two interval matrices: [a,b] + [c,d] = [a+c, b+d]"""
    return np.stack([
        A[..., 0] + B[..., 0],  # Add minimums
        A[..., 1] + B[..., 1]   # Add maximums
    ], axis=-1)

def subtract(A, B):
    """Subtract interval matrices using A - B = A + (-B)"""
    return add(A, negate(B))

def multiply(A, B):
    """Multiply interval matrices: products of all combinations, then take min/max"""
    # have to multiply each combination of min/max
    products = np.array([
        A[..., 0] * B[..., 0],
        A[..., 0] * B[..., 1],
        A[..., 1] * B[..., 0],
        A[..., 1] * B[..., 1]
    ])

    return np.stack([
        np.min(products, axis=0),
        np.max(products, axis=0)
    ], axis=-1)

def intersect_intervals(A, B):
    """Intersect two interval matrices"""
    return np.stack([
        np.maximum(A[..., 0], B[..., 0]),  # Take max of lower bounds
        np.minimum(A[..., 1], B[..., 1])   # Take min of upper bounds
    ], axis=-1)

def _identity_interval(n):
    """Create an interval identity matrix of size n x n"""
    I = np.zeros((n, n, 2))
    for i in range(n):
        I[i, i] = [1, 1]  # Diagonal entries are precise [1,1] intervals
    return I

def _interval_reciprocal(interval):
    """Debug version with printing"""
    a, b = interval
    if a > 0 or b < 0:
        result = np.array([1/b, 1/a]) if a > 0 else np.array([1/a, 1/b])
        return result
    else:
        raise ValueError("Reciprocal is undefined for intervals containing 0.")

def interval_gauss_seidel(A, b, x0, max_iter=10, tol=1e-6):
    """
    Gauss-Seidel iteration using interval arithmetic.

    Args:
        A: Interval matrix (n x n x 2)
        b: Interval vector (n x 2)
        x0: Initial guess (n x 2)
        max_iter: Maximum iterations
        tol: Convergence tolerance
    """
    n = A.shape[0]
    x = x0.copy()

    for _ in range(max_iter):
        x_new = x.copy()
        for i in range(n):
            # Initialize sums as intervals
            s1 = np.zeros((1, 2))  # Sum for j < i
            s2 = np.zeros((1, 2))  # Sum for j > i

            # Compute sum for j < i
            for j in range(i):
                prod = multiply(
                    A[i, j].reshape(1, 1, 2),  # Make 1x1 interval matrix
                    x_new[j].reshape(1, 1, 2)   # Make 1x1 interval matrix
                )
                s1 = add(s1, prod[0])  # Take first (only) row

            # Compute sum for j > i
            for j in range(i + 1, n):
                prod = multiply(
                    A[i, j].reshape(1, 1, 2),  # Make 1x1 interval matrix
                    x[j].reshape(1, 1, 2)      # Make 1x1 interval matrix
                )
                s2 = add(s2, prod[0])  # Take first (only) row

            # Compute right-hand side: b[i] - s1 - s2
            rhs = subtract(
                subtract(b[i].reshape(1, 2), s1),
                s2
            )

            # Multiply by reciprocal of A[i,i]
            recip = _interval_reciprocal(A[i, i])
            prod = multiply(
                rhs.reshape(1, 1, 2),
                recip.reshape(1, 1, 2)
            )
            x_new[i] = prod[0]

        # Intersect with previous x
        x_new = intersect_intervals(x_new, x)

        # Check convergence
        max_change = np.max(np.abs(x_new - x))
        if max_change < tol:
            #print(f"Converged after {it_count + 1} iterations")
            break

        x = x_new

    return x

def inverse(A, max_iter=100, tol=1e-6):
    """
    Compute interval matrix inverse using Gauss-Seidel iteration.

    Args:
        A: Interval matrix (n x n x 2)
        max_iter: Maximum iterations for Gauss-Seidel
        tol: Convergence tolerance

    Returns:
        Inverse interval matrix (n x n x 2)
    """
    n = A.shape[0]

    # Create interval identity matrix
    I = np.zeros((n, n, 2))
    for i in range(n):
        I[i, i] = [1, 1]  # Diagonal entries are [1,1] intervals

    # Initialize result matrix
    X = np.zeros((n, n, 2))

    # Solve AX = I column by column
    for j in range(n):
        # Get j-th column of identity matrix as interval vector
        b = I[:, j]

        # Initial guess (zero intervals)
        x0 = np.zeros((n, 2))

        # Solve for j-th column of X
        X[:, j] = interval_gauss_seidel(A, b, x0, max_iter=max_iter, tol=tol)

    return X
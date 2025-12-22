# Trace Fit Regularization

```{admonition} TODO
:class: warning

This feature is implemented but not currently active. The regularization
parameter is accepted by the order tracing code but has no effect.
```

## Status

The `regularization` parameter in order tracing settings is **not currently used**.

In `trace_orders.py`, the `fit()` function accepts the parameter but uses
`numpy.polynomial.Polynomial.fit()` instead of the regularized `polyfit1d()`:

```python
def fit(x, y, deg, regularization=0):
    # order = polyfit1d(y, x, deg, regularization)  # <-- commented out
    if deg == "best":
        order = best_fit(x, y)
    else:
        order = Polynomial.fit(y, x, deg=deg, domain=[]).coef[::-1]
    return order
```

## Implementation

A Tikhonov regularization implementation exists in `util.py:polyfit1d()`:

```python
def polyfit1d(x, y, degree=1, regularization=0):
    A = np.array([np.power(x, i) for i in idx], dtype=float).T
    b = y.ravel()

    L = np.array([regularization * i**2 for i in idx])
    inv_matrix = np.linalg.inv(A.T @ A + np.diag(L))
    coeff = inv_matrix @ A.T @ b
```

### How it works

Standard least squares minimizes `||Ax - b||^2`, giving solution:

```
x = (A^T A)^{-1} A^T b
```

Tikhonov regularization adds a penalty term `||Lx||^2`:

```
minimize: ||Ax - b||^2 + ||Lx||^2
solution: x = (A^T A + L^T L)^{-1} A^T b
```

The penalty matrix `L` is diagonal with weights `regularization * i^2` where
`i` is the polynomial term index. Higher-degree terms are penalized more
strongly (quadratic 4x, cubic 9x, quartic 16x, etc.).

**Effect:** Smooths the polynomial fit by suppressing high-order wiggles,
preventing overfitting to noise while preserving the overall shape.

## Scaling Issue

The `regularization` parameter is **not normalized**. Its appropriate scale
depends on the magnitude of `A^T A`, which varies with the input data range.

With pixel coordinates (0-4096), `A^T A` has large values for higher polynomial
terms. Useful regularization values might range from `1e-6` to `1e6` depending
on the data - this requires experimentation.

### Recommended fix

Normalize input coordinates before fitting (using the existing `_scale()`
function in `util.py`), which would make the regularization parameter more
intuitive and data-independent (e.g., useful range 0-1).

## TODO

1. Enable regularization by uncommenting the `polyfit1d` call in `trace_orders.fit()`
2. Add input scaling to make the parameter data-independent
3. Document recommended parameter ranges for different use cases
4. Consider whether regularization helps with fiber bundle trace fitting

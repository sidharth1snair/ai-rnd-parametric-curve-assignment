# AI R&D Assignment — Parametric Curve Fit

## Problem

Recover the unknown parameters `θ`, `M`, `X` in the parametric curve:

```
x = t*cos(θ) - e^(M|t|)*sin(0.3t)*sin(θ) + X
y = 42 + t*sin(θ) + e^(M|t|)*sin(0.3t)*cos(θ)
```

given 1500 sample `(x, y)` points known to lie on the curve for `6 < t < 60`,
subject to `0° < θ < 50°`, `-0.05 < M < 0.05`, `0 < X < 100`.

## Result

| Variable | Value |
|---|---|
| θ | 30° (π/6 rad ≈ 0.5236) |
| M | 0.03 |
| X | 55 |

**Desmos / LaTeX expression:**
```
\left(t*\cos(0.5236)-e^{0.03\left|t\right|}\cdot\sin(0.3t)\sin(0.5236)+55,42+t*\sin(0.5236)+e^{0.03\left|t\right|}\cdot\sin(0.3t)\cos(0.5236)\right)
```
Domain: `6 ≤ t ≤ 60`

Desmos graph: **https://www.desmos.com/calculator/qm56f0ggyk**

## Method

### 1. Structural analysis

The equations describe a rotation of a base curve, followed by a translation.
Defining `u(t) = t` and `v(t) = e^(M|t|)*sin(0.3t)`:

```
x - X = u*cos(θ) - v*sin(θ)
y - 42 = u*sin(θ) + v*cos(θ)
```

This is a standard 2D rotation of `(u, v)` by angle `θ`, translated by
`(X, 42)`. So `θ` and `M` fully determine the curve's shape, while `X` is
purely a horizontal shift.



### 2. Handling the absence of a `t` column

The provided `xy_data.csv` contains only `x` and `y` coordinates, with no corresponding `t` values. In addition, the rows are not arranged in the order in which the points were generated. Because of this, it is impossible to determine which `t` value belongs to each data point.

Instead of matching points one by one, the fitting is treated as a **curve-matching problem**. For each candidate set of parameters `(θ, M, X)`, the entire curve is generated, and each data point is matched to the **closest point on that curve**. The overall fit is then evaluated based on these nearest-point distances, without needing to know the original `t` values.




### 3. Loss function — nearest-neighbor (Chamfer) distance

For a candidate `(θ, M, X)`:
1. Densely sample the parametric curve over `t ∈ [6, 60]`.
2. Build a KD-tree over the sampled curve points.
3. For each of the 1500 data points, compute the distance to its nearest
   neighbor on the sampled curve.
4. Sum these distances to obtain the total loss.



### 4. Optimization

- **Global search** — First, a broad search was performed over all possible parameter values using `scipy.optimize.differential_evolution` The curve was sampled with 4,000 points, which is sufficient to explore the entire parameter space and avoid getting stuck in incorrect (local) solutions caused by the oscillating `sin(0.3t)` term.


- **Local refinement** — Next, the best solution from the global search was further refined using `scipy.optimize.minimize` with the Nelder–Mead method. This step used a much finer curve sampling (20,000 points) to improve the accuracy of the final parameter estimates.



### 5. Validation

After refining, the average error between the data points and the model is only about 0.0008, which is extremely small. This tiny error is mainly due to how the curve was sampled, not because the model is inaccurate. The optimization process recovered the parameters θ = 30°, M = 0.03, and X = 55, indicating that these are almost certainly the true values used to generate the data.

## Files

- `code.py` — loads data, runs global search, refines locally, prints final parameters and residual.
- `xy_data.csv` — provided data points.

## Requirements

```
numpy
scipy
```

## Running

```
python code.py
```
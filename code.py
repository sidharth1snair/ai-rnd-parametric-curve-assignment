import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import cKDTree

d = np.loadtxt('xy_data.csv', delimiter=',', skiprows=1)
xd, yd = d[:,0], d[:,1]
N = len(xd)
print("N points:", N)

def curve_xy(theta, M, X, t):
    v = np.exp(M*np.abs(t)) * np.sin(0.3*t)
    x = t*np.cos(theta) - v*np.sin(theta) + X
    y = 42 + t*np.sin(theta) + v*np.cos(theta)
    return x, y

def objective(params, t_dense):
    theta, M, X = params
    cx, cy = curve_xy(theta, M, X, t_dense)
    tree = cKDTree(np.column_stack([cx, cy]))
    dist, idx = tree.query(np.column_stack([xd, yd]), k=1)
    return np.sum(dist), idx

def loss(params, t_dense):
    s, _ = objective(params, t_dense)
    return s

if __name__ == '__main__':
    t_dense = np.linspace(6, 60, 4000)

    bounds = [(np.radians(0.01), np.radians(49.99)), (-0.0499, 0.0499), (0.01, 99.99)]

    res = differential_evolution(loss, bounds, args=(t_dense,), maxiter=60, popsize=20, tol=1e-8, seed=0, polish=True, workers=-1)
    print(res.x, res.fun)
    theta, M, X = res.x
    print("theta deg", np.degrees(theta), "M", M, "X", X)

   
    t_dense2 = np.linspace(6, 60, 20000)
    res2 = minimize(loss, res.x, args=(t_dense2,), method='Nelder-Mead',
                     options={'xatol':1e-10,'fatol':1e-12,'maxiter':5000})
    theta2, M2, X2 = res2.x
    print("refined:", np.degrees(theta2), M2, X2, res2.fun)

    s, idx = objective(res2.x, t_dense2)
    print("mean residual per point:", s/N)
    print("max residual:", None)

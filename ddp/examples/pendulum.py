import sympy as sym
import numpy as np
from time import time

from ddp import DDPOptimizer

with_plots = False
try:
    import matplotlib.pyplot as plt

    with_plots = True
except ImportError:
    print("ERROR: matplotlib not found. Skipping plots")

# dynamics parameters
G = 9.80665
M = 1.0
L = 1.0
dt = 0.05

# dynamics
def f(x, u, constrain=True):
    theta = sym.atan2(x[0], x[1])
    theta_dot = x[2]
    torque = sym.tanh(u[0]) if constrain else u[0]
    theta_dot_dot = -3 * G * sym.sin(theta + sym.pi) / (2 * L) + 3 * torque / (
        M * L**2
    )
    theta += theta_dot * dt
    theta_dot += theta_dot_dot * dt
    return sym.Matrix([sym.sin(theta), sym.cos(theta), theta_dot])


# instantenious cost
def g(x, u, x_goal):
    error = x - x_goal
    Q = np.array([[L**2, L, 0.0], [L, L**2, 0.0], [0.0, 0.0, 0.1]])
    R = np.array([[0.3]])
    result = error.T @ Q @ error + u.T @ R @ u
    return result


# termination cost
def h(x, x_goal):
    error = x - x_goal
    Qt = 100 * np.eye(3)
    result = error.T @ Qt @ error
    return result


# trajectory parameters
N = 100  # trajectory points
Nx = 3  # state dimension
Nu = 1  # control dimesions

# starting state
x0 = np.array([np.sin(np.pi), np.cos(np.pi), 0.0])

# goal state we want to reach
x_goal = np.array([np.sin(0.0), np.cos(0.0), 0.0])

print("Starting state", x0)
print("Goal state", x_goal)

# Create and run optimizer with random intialization
print("Starting optimization")
start_time = time()
ddp = DDPOptimizer(Nx, Nu, f, g, h)
X, U, X_hist, U_hist, J_hist = ddp.optimize(x0, x_goal, N=N, full_output=True)
print("Finished optimization in {:.2f}s".format(time() - start_time))

# plot results
if with_plots:
    print("Plotting results")

    fig, ax = plt.subplots(3, 1, figsize=(4, 8))
    tt = np.linspace(0, dt * N, N)
    theta_sol = np.unwrap(np.arctan2(X[:, 0], X[:, 1]))
    theta_dot_sol = X[:, 2]

    ax[0].plot(theta_sol, theta_dot_sol)
    ax[0].set_xlabel(r"$\theta (rad)$")
    ax[0].set_ylabel(r"$\dot{\theta} (rad/s)$")
    ax[0].set_title("Phase Plot")
    ax[1].set_title("Control")
    ax[1].plot(tt, np.tanh(U))
    ax[1].set_xlabel("Time (s)")
    ax[2].plot(J_hist)
    ax[2].set_title("Trajectory cost")
    ax[2].set_xlabel("Iteration")
    plt.tight_layout()
    plt.savefig("ddp_pendulum.png")

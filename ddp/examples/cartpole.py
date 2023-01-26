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
mp = 0.1
mc = 1.0
l = 1.0
G = 9.80665
dt = 0.05

# dynamics
def f(x, u, constrain=True):

    x_ = x[0]
    x_dot = x[1]
    sin_theta = x[2]
    cos_theta = x[3]
    theta_dot = x[4]
    F = sym.tanh(u[0]) if constrain else u[0]

    # Define dynamics model as per Razvan V. Florian's
    # "Correct equations for the dynamics of the cart-pole system".
    # Friction is neglected.

    # Eq. (23)
    temp = (F + mp * l * theta_dot**2 * sin_theta) / (mc + mp)
    numerator = G * sin_theta - cos_theta * temp
    denominator = l * (4.0 / 3.0 - mp * cos_theta**2 / (mc + mp))
    theta_dot_dot = numerator / denominator

    # Eq. (24)
    x_dot_dot = temp - mp * l * theta_dot_dot * cos_theta / (mc + mp)

    # Deaugment state for dynamics.
    theta = sym.atan2(sin_theta, cos_theta)
    next_theta = theta + theta_dot * dt

    return sym.Matrix(
        [
            x_ + x_dot * dt,
            x_dot + x_dot_dot * dt,
            sym.sin(next_theta),
            sym.cos(next_theta),
            theta_dot + theta_dot_dot * dt,
        ]
    )


# instantenious cost
def g(x, u, x_goal):
    error = x - x_goal
    Q = np.eye(len(x))
    Q[1, 1] = Q[4, 4] = 0.0
    R = 0.1 * np.eye(len(u))
    return error.T @ Q @ error + u.T @ R @ u


# termination cost
def h(x, x_goal):
    error = x - x_goal
    Q = 100 * np.eye(len(x))
    return error.T @ Q @ error


# trajectory parameters
N = 100  # trajectory points
Nx = 5  # state dimension
Nu = 1  # control dimesions

# starting state
x0 = np.array([0.0, 0.0, np.sin(np.pi), np.cos(np.pi), 0.0])

# goal state we want to reach
x_goal = np.array([0.0, 0.0, np.sin(0.0), np.cos(0.0), 0.0])

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
    theta_sol = np.unwrap(np.arctan2(X[:, 2], X[:, 3]))
    theta_dot_sol = X[:, 4]

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
    plt.savefig("ddp_cartpole.pdf")

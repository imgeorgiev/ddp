from time import time
import logging
import numpy as np
from numpy.linalg import inv
import sympy as sym
from sympy import lambdify

from collections.abc import Callable
from numpy.typing import ArrayLike


class DDPOptimizer:
    """Finite horizon Discrete-time Differential Dynamic Programming(DDP)"""

    def __init__(
        self,
        Nx: int,
        Nu: int,
        dynamics: Callable,
        inst_cost: Callable,
        terminal_cost: Callable,
        tolerance: float = 1e-5,
        max_iters: int = 200,
        with_hessians: bool = False,
        constrain: bool = False,
        alphas: ArrayLike = [1.0],
    ):
        """
        Instantiates a DDP Optimizer and pre-computes the dynamics
        and cost derivates without doing any optimization/solving.

        :param Nx: dimension of the state variable x
        :param Nu: dimension of the control variable u
        :param dynamics: a callable dynamics function with 3 arguments
            x, u, constrain. This function must be closed-form differentiable
            by sympy. In other words, it must be built with sympy and/or numpy.
            Has to return the next state x' with same dimensions as input state.
        :param inst_cost: instantenious (aka running) cost funciton. Must be a
            callable function with 3 arguments x, u, x_goal. Again, must be
            closed-form differentiatable by sympy.
        :param term_cost: terminal cost funciton. Must be a
            callable function with 3 arguments x, u, x_goal. Again, must be
            closed-form differentiatable by sympy.
        :param tolerance: tolerance for convergence. Since DDP does multiple
            runs to optimize a trajectory, we add a tolerance at which optimizing
            further doesn't gain us benfits.
        :param max_iters: maximum number of optimization iterations to perform
        :param with_hessians: if true, does the complete DDP optimization including
            the hessians of the dynamics. This is the complete form of the original
            algorithm but often the computational cost of computing these hessians
            are not worth convergence rate increase.
        :param constrain: whether to constrain the dynamics
        :param alphas: list of backtracking coefficients. Must be <=0 and in
            decreasing order. The length of the list designates the number
            of backtracking attempts. A common configuration is
            `1.1 ** (-np.arange(10) ** 2)`. Backtracking often helps but
            is not necessary to obtain good solutions. To not use backtracking
            just leave at default [1.0]
        """
        assert tolerance > 0
        assert max_iters > 0

        self.Nx = Nx
        self.Nu = Nu
        self.tolerance = tolerance
        self.max_iters = max_iters
        self.with_hessians = with_hessians
        self.constrain = constrain
        self.alphas = alphas

        # Pre-compute derivatives now so that we don't have to do it every time
        x = sym.symbols("x:{:}".format(Nx))
        x = sym.Matrix([xi for xi in x])
        u = sym.symbols("u:{:}".format(Nu))
        u = sym.Matrix([ui for ui in u])
        x_goal = sym.symbols("x_g:{:}".format(Nx))
        x_goal = sym.Matrix([xi for xi in x_goal])

        # dynamics
        self.f = lambdify((x, u), dynamics(x, u, constrain))
        self.fx = lambdify((x, u), dynamics(x, u, constrain).jacobian(x))
        self.fu = lambdify((x, u), dynamics(x, u, constrain).jacobian(u))
        jac = dynamics(x, u).jacobian(x)
        self.fxx = lambdify((x, u), [jac.row(i).jacobian(x) for i in range(Nx)])
        jac = dynamics(x, u).jacobian(u)
        self.fux = lambdify((x, u), [jac.row(i).jacobian(x) for i in range(Nx)])
        jac = dynamics(x, u).jacobian(u)
        self.fuu = lambdify((x, u), [jac.row(i).jacobian(u) for i in range(Nx)])

        # costs
        self.g = lambdify((x, u, x_goal), inst_cost(x, u, x_goal))
        self.gx = lambdify((x, u, x_goal), inst_cost(x, u, x_goal).jacobian(x))
        self.gu = lambdify((x, u, x_goal), inst_cost(x, u, x_goal).jacobian(u))
        self.gxx = lambdify(
            (x, u, x_goal), inst_cost(x, u, x_goal).jacobian(x).jacobian(x)
        )
        self.gux = lambdify(
            (x, u, x_goal), inst_cost(x, u, x_goal).jacobian(u).jacobian(x)
        )
        self.guu = lambdify(
            (x, u, x_goal), inst_cost(x, u, x_goal).jacobian(u).jacobian(u)
        )
        self.h = lambdify((x, x_goal), terminal_cost(x, x_goal))
        self.hx = lambdify((x, x_goal), terminal_cost(x, x_goal).jacobian(x))
        self.hxx = lambdify(
            (x, x_goal), terminal_cost(x, x_goal).jacobian(x).jacobian(x)
        )

    def optimize(
        self,
        x0: ArrayLike,
        x_goal: ArrayLike,
        N: int = None,
        U0: ArrayLike = None,
        full_output: bool = False,
    ):
        """
        Optimize a trajectory given a starting state and a goal state.
        Optimization is performed until convergence or until we run out
        of the maximum number of iterations. If the latter happesn, that
        means that the trajectory is suboptimal and there likely is something
        wrongly configured.
        Note that the lenght of the trajectory is decided based on the args.

        :param x0: starting state. Must be of dimensions (Nx,1)
        :param x_goal: goal state. Must be of dimensions (Nx,1)
        :param N: trajectory lenght. If provided, the optimizer generates
            a random initial control sequence. This is called "slow start"
            and often results in poor optimization time (>1s)
        :param U0: initial control sequence. Must be of dimensions (N,Nu)
            where the N is the implied trajectory lenght. If provided this
            will be used to "warm start" the optimization, resulting in
            faster convergence rates (if the warm start is good)
        :param full_output: By default this function returns only the
            optimal state and control sequences. If full_output=True
            it also returns (optimal state sequence, optimal control sequence,
            state sequence history, control sequence history, total cost history)
        """

        if not N and not U0:
            print(
                "ERROR: You have to provide either trajectory length N or initial control sequency U0"
            )
            return

        start = time()
        x0 = np.array(x0)
        x_goal = np.array(x_goal)
        done = False

        # figure out initial control sequence
        if U0:
            N = len(U0) + 1
            U = np.array(U0)
        else:
            assert N > 0
            U = np.random.uniform(-1.0, 1.0, (N, self.Nu))

        # Definte total cost of trajectory function
        # Note: defined here to give the flexibility of parameteraising
        #   x_goal on the fly
        def J(X, U):
            total_cost = 0.0
            for i in range(len(U)):
                total_cost += self.g(X[i], U[i], x_goal)
            total_cost += self.h(X[-1], x_goal)
            return float(total_cost)

        # rollout initial trajectory
        X = np.zeros((N + 1, self.Nx))
        X[0] = x0
        for i in range(len(U)):
            X[i + 1] = self.f(X[i], U[i]).flatten()

        last_cost = J(X, U)

        # keep a history of the trajectory
        if full_output:
            X_hist = [X.copy()]
            U_hist = [U.copy()]
            cost_hist = [last_cost]

        # Start optimization
        logging.info("Starting DDP optimization with J={:.2f}".format(last_cost))
        for i in range(self.max_iters):

            # Backwards pass
            Vx = self.hx(X[-1], x_goal).flatten()
            assert Vx.shape == (self.Nx,)
            Vxx = self.hxx(X[-1], x_goal)
            assert Vxx.shape == (self.Nx, self.Nx)

            # Create buffers for Q derivatives
            Qus = np.zeros((N, self.Nu))
            Quus = np.zeros((N, self.Nu, self.Nu))
            Quxs = np.zeros((N, self.Nu, self.Nx))
            for t in reversed(range(N)):
                gx = self.gx(X[t], U[t], x_goal).flatten()
                assert gx.shape == (self.Nx,)
                gu = self.gu(X[t], U[t], x_goal).flatten()
                assert gu.shape == (self.Nu,)
                gxx = self.gxx(X[t], U[t], x_goal)
                assert gxx.shape == (self.Nx, self.Nx)
                gux = self.gux(X[t], U[t], x_goal)
                assert gux.shape == (self.Nu, self.Nx)
                guu = self.guu(X[t], U[t], x_goal)
                assert guu.shape == (self.Nu, self.Nu)
                fx = self.fx(X[t], U[t])
                assert fx.shape == (self.Nx, self.Nx)
                fu = self.fu(X[t], U[t])
                assert fu.shape == (self.Nx, self.Nu)

                if self.with_hessians:
                    fxx_e = np.array(self.fxx(X[t], U[t]))
                    assert fxx_e.shape == (self.Nx, self.Nx, self.Nx)
                    fux_e = np.array(self.fux(X[t], U[t]))
                    assert fux_e.shape == (self.Nx, self.Nu, self.Nx)
                    fuu_e = np.array(self.fuu(X[t], U[t]))
                    assert fuu_e.shape == (self.Nx, self.Nu, self.Nu)

                Qx = gx + fx.T @ Vx
                assert Qx.shape == (self.Nx,)
                Qu = gu + fu.T @ Vx
                assert Qu.shape == (self.Nu,)
                Qxx = gxx + fx.T @ Vxx @ fx
                assert Qxx.shape == (self.Nx, self.Nx)
                Quu = guu + fu.T @ Vxx @ fu
                assert Quu.shape == (self.Nu, self.Nu)
                Qux = gux + fu.T @ Vxx @ fx
                assert Qux.shape == (self.Nu, self.Nx)

                if self.with_hessians:
                    Qxx += np.tensordot(Vx, fxx_e, axes=1)
                    Quu += np.tensordot(Vx, fuu_e, axes=1)
                    Qux += np.tensordot(Vx, fux_e, axes=1)

                # store Q derivatives for forward pass
                Qus[t] = Qu
                Quus[t] = Quu
                Quxs[t] = Qux

                Quu_inv = inv(Quu)
                Vx = Qx - Qux.T @ Quu_inv @ Qu
                assert Vx.shape == (self.Nx,)
                Vxx = Qxx - Qux.T @ Quu_inv @ Qux
                assert Vxx.shape == (self.Nx, self.Nx)

            # forward pass with backtracking
            for k, alpha in enumerate(self.alphas):
                X_star = np.zeros_like(X)
                U_star = np.zeros_like(U)
                X_star[0] = X[0].copy()
                for t in range(N):
                    error = X_star[t] - X[t]
                    U_star[t] = U[t] - inv(Quus[t]) @ (alpha * Qus[t] + Quxs[t] @ error)
                    X_star[t + 1] = self.f(X_star[t], U_star[t]).flatten()

                # update cost metric to see if we're doing well
                total_cost = J(X_star, U_star)
                if total_cost < last_cost:
                    logging.info(
                        "Accepting new solution with J={:} alpha={:.2f} and {:} backtracks".format(
                            total_cost, alpha, k
                        )
                    )
                    X = X_star
                    U = U_star
                    break

                if alpha == self.alphas[-1]:
                    logging.warn("Reached final alpha")
                    done = True

            if full_output:
                X_hist.append(X.copy())
                U_hist.append(U.copy())
                cost_hist.append(total_cost)

            # check for convergence at the end of the optimization cylcle
            if done or abs(last_cost - total_cost) < self.tolerance:
                break
            last_cost = total_cost

        time_taken = time() - start
        logging.info("Converged in {:}/{:} iterations".format(i, self.max_iters))
        logging.info("Total optimzation time {:.2f}".format(time_taken))
        logging.info("Final trajectory cost J={:.2f}".format(total_cost))

        if full_output:
            return X, U, X_hist, U_hist, cost_hist

        return X, U

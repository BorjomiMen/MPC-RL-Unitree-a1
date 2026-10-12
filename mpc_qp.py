import numpy as np
import osqp
from scipy import sparse


class StandingMPC:
    def __init__(
        self, Ad, Bd, cd, x_ref, u_ref,
        horizon=15, mu=0.5, fz_max=100.0,
    ):
        if horizon < 1 or mu <= 0 or fz_max <= 0:
            raise ValueError("Invalid horizon or contact limits")

        nx, nu = 12, 12
        N = horizon
        self.nx = nx
        self.nu = nu
        self.control_offset = (N + 1) * nx
        x_ref = np.asarray(x_ref, dtype=float).reshape(nx).copy()
        u_ref = np.asarray(u_ref, dtype=float).reshape(nu).copy()
        cd = np.asarray(cd, dtype=float).reshape(nx)
        Ad = sparse.csc_matrix(Ad)
        Bd = sparse.csc_matrix(Bd)
        if Ad.shape != (nx, nx) or Bd.shape != (nx, nu):
            raise ValueError("Expected 12-state, 12-input dynamics")

        Q = sparse.diags(
            [120, 120, 30, 20, 20, 250, 2, 2, 1, 5, 5, 15],
            format="csc", dtype=float,
        )
        QN = 5.0 * Q
        R = 0.001 * sparse.eye(nu, format="csc")
        P = sparse.block_diag([
            sparse.kron(sparse.eye(N), 2.0 * Q),
            2.0 * QN,
            sparse.kron(sparse.eye(N), 2.0 * R),
        ], format="csc")
        q = np.concatenate([
            np.tile(-2.0 * (Q @ x_ref), N),
            -2.0 * (QN @ x_ref),
            np.tile(-2.0 * (R @ u_ref), N),
        ])

        Ax = (
            sparse.kron(sparse.eye(N + 1), -sparse.eye(nx))
            + sparse.kron(sparse.eye(N + 1, k=-1), Ad)
        )
        input_steps = sparse.vstack([
            sparse.csc_matrix((1, N)), sparse.eye(N),
        ], format="csc")
        Bu = sparse.kron(input_steps, Bd)
        Aeq = sparse.hstack([Ax, Bu], format="csc")
        eq_bounds = np.concatenate([-x_ref, -np.tile(cd, N)])

        # An inscribed friction pyramid: ||f_xy|| <= mu * f_z.
        mu_axis = mu / np.sqrt(2.0)
        Cfoot = sparse.csc_matrix([
            [1, 0, -mu_axis],
            [-1, 0, -mu_axis],
            [0, 1, -mu_axis],
            [0, -1, -mu_axis],
            [0, 0, 1],
        ])
        Cforces = sparse.kron(sparse.eye(4 * N), Cfoot, format="csc")
        Aforces = sparse.hstack([
            sparse.csc_matrix((20 * N, self.control_offset)),
            Cforces,
        ], format="csc")
        force_lower = np.tile([-np.inf, -np.inf, -np.inf, -np.inf, 0], 4 * N)
        force_upper = np.tile([0, 0, 0, 0, fz_max], 4 * N)
        self.lower = np.concatenate([eq_bounds, force_lower])
        self.upper = np.concatenate([eq_bounds, force_upper])
        A = sparse.vstack([Aeq, Aforces], format="csc")

        self.solver = osqp.OSQP()
        self.solver.setup(
            P=P, q=q, A=A, l=self.lower, u=self.upper,
            verbose=False, warm_starting=True, polishing=True,
            eps_abs=1e-5, eps_rel=1e-5, max_iter=5000,
        )
        self.last_info = None

    def solve(self, x):
        x = np.asarray(x, dtype=float).reshape(self.nx)
        if not np.isfinite(x).all():
            raise ValueError("MPC state contains non-finite values")
        self.lower[:self.nx] = -x
        self.upper[:self.nx] = -x
        self.solver.update(l=self.lower, u=self.upper)
        result = self.solver.solve(raise_error=False)
        self.last_info = result.info
        if result.info.status != "solved":
            raise RuntimeError(f"MPC solve failed: {result.info.status}")
        start = self.control_offset
        forces = result.x[start:start + self.nu].reshape(4, 3).copy()
        if not np.isfinite(forces).all():
            raise RuntimeError("MPC returned non-finite forces")
        return forces, result.info.status

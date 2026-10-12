
import numpy as np

from mpc_qp import StandingMPC


class ContactMPC(StandingMPC):
    def __init__(
        self, Ad, Bd, cd, x_ref, u_ref, vertical_limits,
        horizon=15, mu=0.5, fz_max=100.0,
    ):
        super().__init__(
            Ad, Bd, cd, x_ref, u_ref,
            horizon=horizon, mu=mu, fz_max=fz_max,
        )
        limits = np.asarray(vertical_limits, dtype=float)
        if limits.shape != (horizon, 4):
            raise ValueError("Expected vertical_limits with shape (horizon, 4)")
        if not np.isfinite(limits).all() or np.any(limits < 0):
            raise ValueError("Contact limits must be finite and nonnegative")
        force_offset = (horizon + 1) * self.nx
        rows = force_offset + np.arange(4 * horizon) * 5 + 4
        self.upper[rows] = np.minimum(limits.ravel(), fz_max)

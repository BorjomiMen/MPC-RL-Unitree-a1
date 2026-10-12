import numpy as np


def skew(v):
    x, y, z = v
    return np.array([
        [0, -z, y],
        [z, 0, -x],
        [-y, x, 0],
    ])


def build_mpc_model(mass, inertia, contacts, com, gravity, dt=0.02):
    Ac = np.zeros((12, 12))
    Ac[0:3, 6:9] = np.eye(3)
    Ac[3:6, 9:12] = np.eye(3)

    Bc = np.zeros((12, 12))
    for i, point in enumerate(contacts):
        columns = slice(3 * i, 3 * i + 3)
        Bc[6:9, columns] = np.linalg.solve(
            inertia, skew(point - com)
        )
        Bc[9:12, columns] = np.eye(3) / mass

    c = np.zeros(12)
    c[9:12] = gravity

    Ad = np.eye(12) + dt * Ac
    Bd = dt * Bc + 0.5 * dt**2 * (Ac @ Bc)
    cd = dt * c + 0.5 * dt**2 * (Ac @ c)

    return Ad, Bd, cd
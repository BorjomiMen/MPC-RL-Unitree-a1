import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from mpc_model import build_mpc_model
from mpc_qp import StandingMPC

path = Path(__file__).parent / "models/unitree_a1/scene_mpc.xml"
model = mujoco.MjModel.from_xml_path(str(path))
model.opt.timestep = 0.001
data = mujoco.MjData(model)

home = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
mujoco.mj_resetDataKeyframe(model, data, home)

joint_ids = model.actuator_trnid[:, 0]
q_indices = model.jnt_qposadr[joint_ids]
v_indices = model.jnt_dofadr[joint_ids]
q_target = data.qpos[q_indices].copy()

mujoco.mj_forward(model, data)
trunk_id = model.body("trunk").id
mass = model.body_subtreemass[trunk_id]
com = data.subtree_com[trunk_id].copy()
inertia = np.zeros((3, 3))

for body_id in np.flatnonzero(model.body_rootid == trunk_id):
    rotation = data.ximat[body_id].reshape(3, 3)
    offset = data.xipos[body_id] - com

    inertia += (
        rotation
        @ np.diag(model.body_inertia[body_id])
        @ rotation.T
    )
    inertia += model.body_mass[body_id] * (
        np.dot(offset, offset) * np.eye(3)
        - np.outer(offset, offset)
    )
feet = []
foot_geom_ids = []
for leg in ("FR", "FL", "RR", "RL"):
    
    calf_id = model.body(f"{leg}_calf").id
    foot_ids = np.flatnonzero(
        (model.geom_bodyid == calf_id)
        & (model.geom_type == mujoco.mjtGeom.mjGEOM_SPHERE)
    )
    feet.append(data.geom_xpos[foot_ids[0]].copy())
    foot_geom_ids.append(int(foot_ids[0]))

r = np.array(feet) - com

# Баланс вертикальных сил и моментов по двум горизонтальным осям.
A = np.vstack((np.ones(4), r[:, 1], -r[:, 0]))
b = np.array([-mass * model.opt.gravity[2], 0.0, 0.0])

fz = np.linalg.lstsq(A, b, rcond=None)[0]
assert np.all(fz >= 0), "Получены отрицательные силы опоры"

print("Силы FR, FL, RR, RL, Н:", np.round(fz, 2))
print("Сумма сил, Н:", fz.sum())
forces = np.zeros((4, 3))
forces[:, 2] = fz
contacts = np.array(feet)
contacts[:, 2] -= model.geom_size[foot_geom_ids, 0]

Ad, Bd, cd = build_mpc_model(
    mass, inertia, contacts, com, model.opt.gravity
)

x0 = np.zeros(12)
x0[3:6] = com

x1 = Ad @ x0 + Bd @ forces.ravel() + cd
print("Ошибка равновесия:", np.max(np.abs(x1 - x0)))
mpc = StandingMPC(Ad, Bd, cd, x0, forces.ravel())

forces_test, status = mpc.solve(x0)
print("Статус MPC:", status)
print("Силы MPC, Н:")
print(np.round(forces_test, 2))
jacp = np.zeros((3, model.nv))
jacr = np.zeros((3, model.nv))
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        start = time.perf_counter()

        mujoco.mj_forward(model, data)

        torque = data.qfrc_bias[v_indices].copy()

        for geom_id, force in zip(foot_geom_ids, forces):
            mujoco.mj_jacGeom(model, data, jacp, jacr, geom_id)
            torque -= jacp[:, v_indices].T @ force

        q = data.qpos[q_indices]
        dq = data.qvel[v_indices]
        torque += 100.0 * (q_target - q) - 3.0 * dq

        data.ctrl[:] = np.clip(
            torque,
            model.actuator_ctrlrange[:, 0],
            model.actuator_ctrlrange[:, 1],
        )

        mujoco.mj_step(model, data)
        viewer.sync()

        remaining = model.opt.timestep - (time.perf_counter() - start)
        if remaining > 0:
            time.sleep(remaining)
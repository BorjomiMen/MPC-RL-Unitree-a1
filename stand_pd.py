import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

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

feet = []
for leg in ("FR", "FL", "RR", "RL"):
    calf_id = model.body(f"{leg}_calf").id
    foot_ids = np.flatnonzero(
        (model.geom_bodyid == calf_id)
        & (model.geom_type == mujoco.mjtGeom.mjGEOM_SPHERE)
    )
    feet.append(data.geom_xpos[foot_ids[0]].copy())

r = np.array(feet) - com

# Баланс вертикальных сил и моментов по двум горизонтальным осям.
A = np.vstack((np.ones(4), r[:, 1], -r[:, 0]))
b = np.array([-mass * model.opt.gravity[2], 0.0, 0.0])

fz = np.linalg.lstsq(A, b, rcond=None)[0]
assert np.all(fz >= 0), "Получены отрицательные силы опоры"

print("Силы FR, FL, RR, RL, Н:", np.round(fz, 2))
print("Сумма сил, Н:", fz.sum())
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        start = time.perf_counter()

        q = data.qpos[q_indices]
        dq = data.qvel[v_indices]
        torque = 100.0 * (q_target - q) - 3.0 * dq

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
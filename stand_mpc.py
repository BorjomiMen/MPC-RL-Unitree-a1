import argparse
from contextlib import nullcontext
from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np

from mpc_model import build_mpc_model
from mpc_qp import StandingMPC


class StandingSimulation:
    def __init__(self, model_path):
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.model.opt.timestep = 0.001
        self.data = mujoco.MjData(self.model)
        model, data = self.model, self.data
        home = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if home < 0:
            raise ValueError("The model needs a home keyframe")
        mujoco.mj_resetDataKeyframe(model, data, home)
        joint_ids = model.actuator_trnid[:, 0]
        self.q_indices = model.jnt_qposadr[joint_ids]
        self.v_indices = model.jnt_dofadr[joint_ids]
        self.q_target = data.qpos[self.q_indices].copy()
        self.trunk_id = model.body("trunk").id
        self.velocity = np.zeros(6)

        mujoco.mj_forward(model, data)
        mass = model.body_subtreemass[self.trunk_id]
        com = data.subtree_com[self.trunk_id].copy()
        inertia = np.zeros((3, 3))
        for body_id in np.flatnonzero(model.body_rootid == self.trunk_id):
            rotation = data.ximat[body_id].reshape(3, 3)
            offset = data.xipos[body_id] - com
            inertia += rotation @ np.diag(model.body_inertia[body_id]) @ rotation.T
            inertia += model.body_mass[body_id] * (
                np.dot(offset, offset) * np.eye(3) - np.outer(offset, offset)
            )

        self.foot_geom_ids = []
        for leg in ("FR", "FL", "RR", "RL"):
            calf_id = model.body(f"{leg}_calf").id
            ids = np.flatnonzero(
                (model.geom_bodyid == calf_id)
                & (model.geom_type == mujoco.mjtGeom.mjGEOM_SPHERE)
            )
            if len(ids) != 1:
                raise ValueError(f"Expected one spherical foot for {leg}")
            self.foot_geom_ids.append(int(ids[0]))

        contacts = data.geom_xpos[self.foot_geom_ids].copy()
        contacts[:, 2] -= model.geom_size[self.foot_geom_ids, 0]
        lever = contacts - com
        balance = np.vstack((np.ones(4), lever[:, 1], -lever[:, 0]))
        target = np.array([-mass * model.opt.gravity[2], 0.0, 0.0])
        fz = np.linalg.lstsq(balance, target, rcond=None)[0]
        if np.any(fz < 0):
            raise ValueError("The nominal stance needs negative support forces")
        self.forces = np.zeros((4, 3))
        self.forces[:, 2] = fz

        self.x_ref = self.read_state()
        Ad, Bd, cd = build_mpc_model(
            mass, inertia, contacts, com, model.opt.gravity, dt=0.02
        )
        self.mpc = StandingMPC(Ad, Bd, cd, self.x_ref, self.forces.ravel())
        self.jacp = np.zeros((3, model.nv))
        self.jacr = np.zeros((3, model.nv))
        self.step_count = 0
        self.steps_per_mpc = 20
        self.mpc_updates = 0
        self.state = self.x_ref.copy()
        self.status = "not solved"
        self.unclipped_torque = np.zeros(model.nu)

    def read_state(self):
        model, data = self.model, self.data
        mujoco.mj_subtreeVel(model, data)
        rotation = data.xmat[self.trunk_id].reshape(3, 3)
        x = np.zeros(12)
        x[:3] = [
            np.arctan2(rotation[2, 1], rotation[2, 2]),
            np.arcsin(np.clip(-rotation[2, 0], -1.0, 1.0)),
            np.arctan2(rotation[1, 0], rotation[0, 0]),
        ]
        x[3:6] = data.subtree_com[self.trunk_id]
        mujoco.mj_objectVelocity(
            model, data, mujoco.mjtObj.mjOBJ_BODY,
            self.trunk_id, self.velocity, 0,
        )
        x[6:9] = self.velocity[:3]
        x[9:12] = data.subtree_linvel[self.trunk_id]
        return x

    def step(self):
        model, data = self.model, self.data
        mujoco.mj_forward(model, data)
        if self.step_count % self.steps_per_mpc == 0:
            self.state = self.read_state()
            self.forces, self.status = self.mpc.solve(self.state)
            self.mpc_updates += 1

        torque = data.qfrc_bias[self.v_indices].copy()
        for geom_id, force in zip(self.foot_geom_ids, self.forces):
            # Use the same ground-facing foot point as the prediction model.
            point = data.geom_xpos[geom_id].copy()
            point[2] -= model.geom_size[geom_id, 0]
            mujoco.mj_jac(
                model, data, self.jacp, self.jacr,
                point, int(model.geom_bodyid[geom_id]),
            )
            torque -= self.jacp[:, self.v_indices].T @ force

        torque += (
            100.0 * (self.q_target - data.qpos[self.q_indices])
            - 3.0 * data.qvel[self.v_indices]
        )
        self.unclipped_torque = torque.copy()
        data.ctrl[:] = np.clip(
            torque, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
        )
        mujoco.mj_step(model, data)
        self.step_count += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seconds", type=float)
    args = parser.parse_args()
    if args.seconds is not None and args.seconds <= 0:
        parser.error("--seconds must be positive")
    duration = args.seconds
    if args.headless and duration is None:
        duration = 30.0

    model_path = Path(__file__).parent / "models/unitree_a1/scene_mpc.xml"
    sim = StandingSimulation(model_path)
    context = (
        nullcontext(None) if args.headless
        else mujoco.viewer.launch_passive(sim.model, sim.data)
    )
    print("Standing MPC: 50 Hz; torque loop: 1000 Hz (simulation time).")
    with context as viewer:
        next_log_time = 1.0
        while duration is None or sim.step_count * sim.model.opt.timestep < duration:
            if viewer is not None and not viewer.is_running():
                break
            start = time.perf_counter()
            sim.step()
            if viewer is not None:
                viewer.sync()
            if sim.data.time >= next_log_time:
                error = sim.state[5] - sim.x_ref[5]
                print(
                    f"t={sim.data.time:.1f}s | MPC={sim.status} | "
                    f"COM z={sim.state[5]:.3f}m | dz={error:+.4f}m"
                )
                next_log_time += 1.0
            if viewer is not None:
                remaining = sim.model.opt.timestep - (time.perf_counter() - start)
                if remaining > 0:
                    time.sleep(remaining)


if __name__ == "__main__":
    main()

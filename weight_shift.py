import argparse
from contextlib import nullcontext
from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np

from mpc_model import build_mpc_model
from mpc_qp import StandingMPC
from stand_mpc import StandingSimulation


def triangle_margin(point, triangle):
    following = np.roll(triangle, -1, axis=0)
    area_twice = np.sum(
        triangle[:, 0] * following[:, 1] - triangle[:, 1] * following[:, 0]
    )
    if abs(area_twice) < 1e-10:
        raise ValueError("Degenerate support triangle")
    orientation = np.sign(area_twice)
    edges = following - triangle
    offsets = point - triangle
    distances = orientation * (
        edges[:, 0] * offsets[:, 1] - edges[:, 1] * offsets[:, 0]
    ) / np.linalg.norm(edges, axis=1)
    return float(np.min(distances))


def robot_inertia(model, data, trunk_id, com):
    inertia = np.zeros((3, 3))
    for body_id in np.flatnonzero(model.body_rootid == trunk_id):
        rotation = data.ximat[body_id].reshape(3, 3)
        offset = data.xipos[body_id] - com
        inertia += rotation @ np.diag(model.body_inertia[body_id]) @ rotation.T
        inertia += model.body_mass[body_id] * (
            np.dot(offset, offset) * np.eye(3) - np.outer(offset, offset)
        )
    return inertia


def foot_ground_loads(sim):
    model, data = sim.model, sim.data
    floor_id = model.geom("floor").id
    leg_bodies = model.geom_bodyid[sim.foot_geom_ids]
    loads = np.zeros(4)
    contact_force = np.zeros(6)
    for index in range(data.ncon):
        contact = data.contact[index]
        if contact.geom1 == floor_id:
            robot_geom, sign = contact.geom2, 1.0
        elif contact.geom2 == floor_id:
            robot_geom, sign = contact.geom1, -1.0
        else:
            continue
        leg = np.flatnonzero(leg_bodies == model.geom_bodyid[robot_geom])
        if not len(leg):
            continue
        mujoco.mj_contactForce(model, data, index, contact_force)
        world_force = contact.frame.reshape(3, 3).T @ contact_force[:3]
        loads[int(leg[0])] += sign * world_force[2]
    return loads


class WeightShiftSimulation(StandingSimulation):
    def __init__(self, model_path):
        super().__init__(model_path)
        self.nominal_com = self.x_ref[3:6].copy()
        self.foot_goals = self.data.geom_xpos[self.foot_geom_ids].copy()
        self.support_indices = [1, 2, 3]  # FL, RR, RL: future three-foot support.
        self.target_com = self.nominal_com.copy()
        self.target_com[:2] = self.foot_goals[self.support_indices, :2].mean(axis=0)
        self.shift_start = 3.0
        self.shift_duration = 3.0
        self.alpha = 0.0
        self.ik_max_error = 0.0
        self.posture_plan = self._plan_postures()

    def _plan_postures(self):
        model = self.model
        pose = mujoco.MjData(model)
        pose.qpos[:] = self.data.qpos
        base_joint = model.body_jntadr[self.trunk_id]
        if model.jnt_type[base_joint] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError("The trunk must have a free joint")
        base_dof = model.jnt_dofadr[base_joint]
        variables = np.r_[np.arange(base_dof, base_dof + 3), self.v_indices]
        joint_limits = model.jnt_range[model.actuator_trnid[:, 0]]
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        jaccom = np.zeros((3, model.nv))
        plan = []
        for alpha in np.linspace(0.0, 1.0, 51):
            desired = self.nominal_com + alpha * (self.target_com - self.nominal_com)
            for _ in range(40):
                mujoco.mj_forward(model, pose)
                error = np.concatenate([
                    (self.foot_goals - pose.geom_xpos[self.foot_geom_ids]).ravel(),
                    desired - pose.subtree_com[self.trunk_id],
                ])
                if np.max(np.abs(error)) < 1e-8:
                    break
                blocks = []
                for geom_id in self.foot_geom_ids:
                    mujoco.mj_jacGeom(model, pose, jacp, jacr, geom_id)
                    blocks.append(jacp[:, variables].copy())
                mujoco.mj_jacSubtreeCom(model, pose, jaccom, self.trunk_id)
                blocks.append(jaccom[:, variables].copy())
                J = np.vstack(blocks)
                delta = J.T @ np.linalg.solve(J @ J.T + 1e-8 * np.eye(15), error)
                delta *= min(1.0, 0.05 / max(float(np.max(np.abs(delta))), 1e-12))
                velocity = np.zeros(model.nv)
                velocity[variables] = delta
                mujoco.mj_integratePos(model, pose.qpos, velocity, 1.0)
                pose.qpos[self.q_indices] = np.clip(
                    pose.qpos[self.q_indices], joint_limits[:, 0], joint_limits[:, 1]
                )
            else:
                raise RuntimeError("Inverse kinematics did not converge")
            self.ik_max_error = max(self.ik_max_error, float(np.max(np.abs(error))))
            plan.append(pose.qpos[self.q_indices].copy())
        return np.array(plan)

    def _update_goals(self):
        model, data = self.model, self.data
        mujoco.mj_forward(model, data)
        phase = np.clip(
            (self.step_count * model.opt.timestep - self.shift_start) / self.shift_duration,
            0.0, 1.0,
        )
        self.alpha = float(phase**3 * (10.0 - 15.0 * phase + 6.0 * phase**2))
        sample = self.alpha * (len(self.posture_plan) - 1)
        index = min(int(sample), len(self.posture_plan) - 2)
        fraction = sample - index
        self.q_target = (
            (1.0 - fraction) * self.posture_plan[index]
            + fraction * self.posture_plan[index + 1]
        )
        self.x_ref[3:6] = self.nominal_com + self.alpha * (
            self.target_com - self.nominal_com
        )

        mass = model.body_subtreemass[self.trunk_id]
        com = data.subtree_com[self.trunk_id].copy()
        inertia = robot_inertia(model, data, self.trunk_id, com)
        contacts = data.geom_xpos[self.foot_geom_ids].copy()
        contacts[:, 2] -= model.geom_size[self.foot_geom_ids, 0]
        lever = contacts - com
        balance = np.vstack((np.ones(4), lever[:, 1], -lever[:, 0]))
        target = np.array([-mass * model.opt.gravity[2], 0.0, 0.0])
        fz = np.linalg.lstsq(balance, target, rcond=None)[0]
        if np.any(fz < 0):
            raise RuntimeError("Current COM needs negative static support forces")
        force_ref = np.zeros((4, 3))
        force_ref[:, 2] = fz
        Ad, Bd, cd = build_mpc_model(
            mass, inertia, contacts, com, model.opt.gravity, dt=0.02
        )
        self.mpc = StandingMPC(Ad, Bd, cd, self.x_ref, force_ref.ravel())

    def step(self):
        if self.step_count % self.steps_per_mpc == 0:
            self._update_goals()
        super().step()

    def future_support_margin(self):
        triangle = self.data.geom_xpos[
            np.array(self.foot_geom_ids)[self.support_indices], :2
        ]
        return triangle_margin(self.data.subtree_com[self.trunk_id, :2], triangle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    model_path = Path(__file__).parent / "models/unitree_a1/scene_mpc.xml"
    sim = WeightShiftSimulation(model_path)
    print("Four-foot weight shift, preparing FR lift. Movement: 3 to 6 s.")
    print("Target COM xyz, m:", np.round(sim.target_com, 4))
    print(f"IK maximum error: {sim.ik_max_error:.2e} m")
    context = (
        nullcontext(None) if args.headless
        else mujoco.viewer.launch_passive(sim.model, sim.data)
    )
    total_steps = round(10.0 / sim.model.opt.timestep)
    max_foot_drift = 0.0
    peak_torque = 0.0
    peak_tilt = 0.0
    clipped_steps = 0
    with context as viewer:
        while sim.step_count < total_steps:
            if viewer is not None and not viewer.is_running():
                break
            start = time.perf_counter()
            sim.step()
            peak_torque = max(peak_torque, float(np.max(np.abs(sim.unclipped_torque))))
            limits = sim.model.actuator_ctrlrange
            clipped_steps += int(np.any(
                (sim.unclipped_torque < limits[:, 0] - 1e-8)
                | (sim.unclipped_torque > limits[:, 1] + 1e-8)
            ))
            if sim.step_count % sim.steps_per_mpc == 1:
                mujoco.mj_forward(sim.model, sim.data)
                drift = np.linalg.norm(
                    sim.data.geom_xpos[sim.foot_geom_ids, :2] - sim.foot_goals[:, :2],
                    axis=1,
                )
                max_foot_drift = max(max_foot_drift, float(np.max(drift)))
                peak_tilt = max(peak_tilt, float(np.max(np.abs(sim.state[:2]))))
            if (sim.step_count - 1) % round(1.0 / sim.model.opt.timestep) == 0:
                print(
                    f"t={sim.data.time:.1f}s | MPC={sim.status} | "
                    f"COM xy={np.round(sim.state[3:5], 4)} | "
                    f"FL/RR/RL margin={sim.future_support_margin() * 1000:.1f}mm"
                )
            if not np.isfinite(sim.data.qpos).all() or sim.data.qpos[2] < 0.15:
                raise RuntimeError("Weight shift failed: non-finite state or low trunk")
            if viewer is not None:
                viewer.sync()
                remaining = sim.model.opt.timestep - (time.perf_counter() - start)
                if remaining > 0:
                    time.sleep(remaining)

    mujoco.mj_forward(sim.model, sim.data)
    final = sim.read_state()
    margin = sim.future_support_margin()
    loads = foot_ground_loads(sim)
    complete = sim.step_count == total_steps
    print("COMPLETE" if complete else "STOPPED EARLY")
    print("Final COM xyz, m:", np.round(final[3:6], 6))
    print(f"Future FL/RR/RL support margin: {margin * 1000:.2f} mm")
    print(f"Maximum foot XY drift: {max_foot_drift * 1000:.2f} mm")
    print(f"Peak roll/pitch: {np.degrees(peak_tilt):.3f} deg")
    print(f"Maximum commanded torque: {peak_torque:.3f} Nm; clipped steps: {clipped_steps}")
    print("Actual vertical ground loads FR/FL/RR/RL, N:", np.round(loads, 2))
    ready = complete and margin > 0.03 and np.all(loads > 1.0) and clipped_steps == 0
    print("WEIGHT SHIFT READY" if ready else "CHECK WEIGHT SHIFT BEFORE CONTINUING")


if __name__ == "__main__":
    main()

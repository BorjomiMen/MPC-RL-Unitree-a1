import argparse
from contextlib import nullcontext
from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np

from contact_mpc import ContactMPC
from mpc_model import build_mpc_model
from weight_shift import WeightShiftSimulation, foot_ground_loads, robot_inertia


def smoothstep(value):
    value = np.clip(value, 0.0, 1.0)
    return float(value**3 * (10.0 - 15.0 * value + 6.0 * value**2))


def interpolate(plan, fraction):
    sample = np.clip(fraction, 0.0, 1.0) * (len(plan) - 1)
    index = min(int(sample), len(plan) - 2)
    weight = sample - index
    return (1.0 - weight) * plan[index] + weight * plan[index + 1]


class FrontRightLiftSimulation(WeightShiftSimulation):
    def __init__(self, model_path):
        super().__init__(model_path)
        self.lift_height = 0.05
        self.unload_start, self.unload_end = 7.0, 8.0
        self.lift_start, self.lift_end = 8.0, 9.0
        self.lower_start, self.lower_end = 11.0, 12.0
        self.reload_start, self.reload_end = 12.5, 13.5
        self.duration = 16.0
        self.horizon = 15
        self.mpc_dt = self.steps_per_mpc * self.model.opt.timestep
        self.fr_stance_limit = float(self.forces[0, 2])
        self.swing_plan = self._plan_lift()
        self.contact_limits = np.full((self.horizon, 4), 100.0)
        self.phase_name = "settle"

    def availability(self, seconds):
        if seconds < self.unload_start:
            return 1.0
        if seconds < self.unload_end:
            return 1.0 - smoothstep(
                (seconds - self.unload_start) / (self.unload_end - self.unload_start)
            )
        if seconds < self.reload_start:
            return 0.0
        return smoothstep(
            (seconds - self.reload_start) / (self.reload_end - self.reload_start)
        )

    def swing_fraction(self, seconds):
        if seconds < self.lift_start:
            return 0.0
        if seconds < self.lift_end:
            return smoothstep(
                (seconds - self.lift_start) / (self.lift_end - self.lift_start)
            )
        if seconds < self.lower_start:
            return 1.0
        return 1.0 - smoothstep(
            (seconds - self.lower_start) / (self.lower_end - self.lower_start)
        )

    def _plan_lift(self):
        model = self.model
        pose = mujoco.MjData(model)
        pose.qpos[:] = self.data.qpos
        pose.qpos[self.q_indices] = self.posture_plan[-1]
        mujoco.mj_forward(model, pose)
        base_joint = model.body_jntadr[self.trunk_id]
        base_qpos = model.jnt_qposadr[base_joint]
        base_dof = model.jnt_dofadr[base_joint]
        pose.qpos[base_qpos:base_qpos + 3] += (
            self.foot_goals[1] - pose.geom_xpos[self.foot_geom_ids[1]]
        )
        variables = np.r_[np.arange(base_dof, base_dof + 3), self.v_indices]
        joint_limits = model.jnt_range[model.actuator_trnid[:, 0]]
        jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        jaccom = np.zeros((3, model.nv))
        plan = []
        for fraction in np.linspace(0.0, 1.0, 51):
            foot_goals = self.foot_goals.copy()
            foot_goals[0, 2] += self.lift_height * fraction
            for _ in range(40):
                mujoco.mj_forward(model, pose)
                error = np.r_[
                    (foot_goals - pose.geom_xpos[self.foot_geom_ids]).ravel(),
                    self.target_com - pose.subtree_com[self.trunk_id],
                ]
                if np.max(np.abs(error)) < 1e-8:
                    break
                blocks = []
                for geom_id in self.foot_geom_ids:
                    mujoco.mj_jacGeom(model, pose, jacp, jacr, geom_id)
                    blocks.append(jacp[:, variables].copy())
                mujoco.mj_jacSubtreeCom(model, pose, jaccom, self.trunk_id)
                blocks.append(jaccom[:, variables].copy())
                jacobian = np.vstack(blocks)
                delta = jacobian.T @ np.linalg.solve(
                    jacobian @ jacobian.T + 1e-8 * np.eye(15), error
                )
                delta *= min(1.0, 0.05 / max(np.max(np.abs(delta)), 1e-12))
                velocity = np.zeros(model.nv)
                velocity[variables] = delta
                mujoco.mj_integratePos(model, pose.qpos, velocity, 1.0)
                pose.qpos[self.q_indices] = np.clip(
                    pose.qpos[self.q_indices], joint_limits[:, 0], joint_limits[:, 1]
                )
            else:
                raise RuntimeError("Swing-foot IK did not converge")
            self.ik_max_error = max(self.ik_max_error, float(np.max(np.abs(error))))
            plan.append(pose.qpos[self.q_indices].copy())
        return np.array(plan)

    def _update_goals(self):
        model, data = self.model, self.data
        mujoco.mj_forward(model, data)
        seconds = self.step_count * model.opt.timestep
        self.alpha = smoothstep((seconds - self.shift_start) / self.shift_duration)
        self.q_target = interpolate(self.posture_plan, self.alpha)
        if seconds >= self.lift_start:
            self.q_target = interpolate(self.swing_plan, self.swing_fraction(seconds))
        self.x_ref[3:6] = self.nominal_com + self.alpha * (
            self.target_com - self.nominal_com
        )
        if seconds < self.shift_start:
            self.phase_name = "settle"
        elif seconds < self.shift_start + self.shift_duration:
            self.phase_name = "shift"
        elif seconds < self.unload_start:
            self.phase_name = "ready"
        elif seconds < self.unload_end:
            self.phase_name = "unload FR"
        elif seconds < self.lift_end:
            self.phase_name = "lift FR"
        elif seconds < self.lower_start:
            self.phase_name = "three feet"
        elif seconds < self.lower_end:
            self.phase_name = "lower FR"
        elif seconds < self.reload_start:
            self.phase_name = "touchdown"
        elif seconds < self.reload_end:
            self.phase_name = "reload FR"
        else:
            self.phase_name = "four feet"

        mass = model.body_subtreemass[self.trunk_id]
        com = data.subtree_com[self.trunk_id].copy()
        inertia = robot_inertia(model, data, self.trunk_id, com)
        contacts = data.geom_xpos[self.foot_geom_ids].copy()
        contacts[:, 2] -= model.geom_size[self.foot_geom_ids, 0]
        lever = contacts - com
        balance = np.vstack((np.ones(4), lever[:, 1], -lever[:, 0]))
        target = np.array([-mass * model.opt.gravity[2], 0.0, 0.0])
        original_fz = np.linalg.lstsq(balance, target, rcond=None)[0]
        availability = self.availability(seconds)
        fz = original_fz.copy()
        fz[0] = max(0.0, original_fz[0]) * availability
        fz[1:] = np.linalg.solve(balance[:, 1:], target - balance[:, 0] * fz[0])
        if np.min(fz) < -1e-6:
            raise RuntimeError("COM is outside the active support polygon")
        force_ref = np.zeros((4, 3))
        force_ref[:, 2] = np.maximum(fz, 0.0)

        self.contact_limits = np.full((self.horizon, 4), 100.0)
        for step in range(self.horizon):
            future = seconds + step * self.mpc_dt
            if future >= self.unload_start:
                self.contact_limits[step, 0] = (
                    self.fr_stance_limit * self.availability(future)
                )
        Ad, Bd, cd = build_mpc_model(
            mass, inertia, contacts, com, model.opt.gravity, dt=self.mpc_dt
        )
        self.mpc = ContactMPC(
            Ad, Bd, cd, self.x_ref, force_ref.ravel(), self.contact_limits,
            horizon=self.horizon,
        )

    def fr_clearance(self):
        geom_id = self.foot_geom_ids[0]
        floor_height = self.data.geom_xpos[self.model.geom("floor").id, 2]
        return float(
            self.data.geom_xpos[geom_id, 2]
            - self.model.geom_size[geom_id, 0] - floor_height
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    model_path = Path(__file__).parent / "models/unitree_a1/scene_mpc.xml"
    sim = FrontRightLiftSimulation(model_path)
    print("FR: unload 7-8 s, lift 8-9 s, hold 9-11 s, lower 11-12 s.")
    print("Return four-foot support: 12.5-13.5 s. Total: 16 s.")
    print(f"IK maximum error: {sim.ik_max_error:.2e} m")
    context = (
        nullcontext(None) if args.headless
        else mujoco.viewer.launch_passive(sim.model, sim.data)
    )
    total_steps = round(sim.duration / sim.model.opt.timestep)
    peak_torque = peak_tilt = max_drift = max_swing_force = 0.0
    clipped_steps = hold_samples = 0
    hold_min_clearance = hold_min_margin = hold_min_support_load = np.inf
    hold_max_fr_load = 0.0
    with context as viewer:
        while sim.step_count < total_steps:
            if viewer is not None and not viewer.is_running():
                break
            started = time.perf_counter()
            sim.step()
            peak_torque = max(peak_torque, float(np.max(np.abs(sim.unclipped_torque))))
            limits = sim.model.actuator_ctrlrange
            clipped_steps += int(np.any(
                (sim.unclipped_torque < limits[:, 0] - 1e-8)
                | (sim.unclipped_torque > limits[:, 1] + 1e-8)
            ))
            if not np.isfinite(sim.data.qpos).all() or sim.data.qpos[2] < 0.15:
                raise RuntimeError("Lift failed: non-finite state or low trunk")
            if sim.step_count % sim.steps_per_mpc == 1:
                mujoco.mj_forward(sim.model, sim.data)
                state = sim.read_state()
                loads = foot_ground_loads(sim)
                clearance = sim.fr_clearance()
                margin = sim.future_support_margin()
                peak_tilt = max(peak_tilt, float(np.max(np.abs(state[:2]))))
                drift = np.linalg.norm(
                    sim.data.geom_xpos[sim.foot_geom_ids, :2][1:]
                    - sim.foot_goals[1:, :2], axis=1,
                )
                max_drift = max(max_drift, float(np.max(drift)))
                seconds = (sim.step_count - 1) * sim.model.opt.timestep
                if sim.unload_end <= seconds < sim.reload_start:
                    max_swing_force = max(max_swing_force, float(np.max(np.abs(sim.forces[0]))))
                if sim.lift_end + 0.3 <= seconds < sim.lower_start:
                    hold_samples += 1
                    hold_min_clearance = min(hold_min_clearance, clearance)
                    hold_min_margin = min(hold_min_margin, margin)
                    hold_min_support_load = min(hold_min_support_load, float(np.min(loads[1:])))
                    hold_max_fr_load = max(hold_max_fr_load, abs(float(loads[0])))
                if (sim.step_count - 1) % round(1.0 / sim.model.opt.timestep) == 0:
                    print(
                        f"t={seconds:.1f}s | {sim.phase_name} | MPC={sim.status} | "
                        f"FR clearance={clearance * 1000:.1f}mm | "
                        f"FR load={loads[0]:.2f}N | margin={margin * 1000:.1f}mm"
                    )
            if viewer is not None:
                viewer.sync()
                remaining = sim.model.opt.timestep - (time.perf_counter() - started)
                if remaining > 0:
                    time.sleep(remaining)

    mujoco.mj_forward(sim.model, sim.data)
    final = sim.read_state()
    loads = foot_ground_loads(sim)
    complete = sim.step_count == total_steps
    print("COMPLETE" if complete else "STOPPED EARLY")
    print("Final COM xyz, m:", np.round(final[3:6], 6))
    if hold_samples:
        print(f"Hold minimum FR clearance: {hold_min_clearance * 1000:.2f} mm")
        print(f"Hold maximum actual FR load: {hold_max_fr_load:.4f} N")
        print(f"Hold minimum FL/RR/RL load: {hold_min_support_load:.2f} N")
        print(f"Hold minimum support margin: {hold_min_margin * 1000:.2f} mm")
    else:
        print("Three-foot hold was not measured")
    print(f"Maximum MPC force component on inactive FR: {max_swing_force:.2e} N")
    print(f"Maximum support foot XY drift: {max_drift * 1000:.2f} mm")
    print(f"Peak roll/pitch: {np.degrees(peak_tilt):.3f} deg")
    print(f"Maximum commanded torque: {peak_torque:.3f} Nm; clipped steps: {clipped_steps}")
    print("Final actual ground loads FR/FL/RR/RL, N:", np.round(loads, 2))
    ready = (
        complete and hold_samples > 0 and hold_min_clearance > 0.025
        and hold_max_fr_load < 0.1 and hold_min_support_load > 5.0
        and hold_min_margin > 0.03 and max_swing_force < 1e-3
        and np.degrees(peak_tilt) < 3.0 and max_drift < 0.01
        and clipped_steps == 0 and np.all(loads > 1.0)
    )
    print("FR LIFT PASSED" if ready else "CHECK FR LIFT BEFORE CONTINUING")


if __name__ == "__main__":
    main()

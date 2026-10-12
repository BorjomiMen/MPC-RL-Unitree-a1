import argparse
from contextlib import nullcontext
import csv
from datetime import datetime
from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np

from contact_mpc import ContactMPC
from lift_fr import interpolate, smoothstep
from mpc_model import build_mpc_model
from weight_shift import WeightShiftSimulation, foot_ground_loads, robot_inertia


class WholeBodyIK:
    """Solve four foot centers and whole-robot COM with fixed base attitude."""

    def __init__(self, sim, initial_qpos):
        self.sim = sim
        self.pose = mujoco.MjData(sim.model)
        self.pose.qpos[:] = initial_qpos
        base_joint = sim.model.body_jntadr[sim.trunk_id]
        base_dof = sim.model.jnt_dofadr[base_joint]
        self.variables = np.r_[np.arange(base_dof, base_dof + 3), sim.v_indices]
        self.limits = sim.model.jnt_range[sim.model.actuator_trnid[:, 0]]
        self.jacp = np.zeros((3, sim.model.nv))
        self.jacr = np.zeros_like(self.jacp)
        self.jaccom = np.zeros_like(self.jacp)
        self.max_error = 0.0

    def solve(self, foot_goals, com_goal):
        sim, pose = self.sim, self.pose
        for _ in range(50):
            mujoco.mj_forward(sim.model, pose)
            error = np.r_[
                (foot_goals - pose.geom_xpos[sim.foot_geom_ids]).ravel(),
                com_goal - pose.subtree_com[sim.trunk_id],
            ]
            if np.max(np.abs(error)) < 1e-8:
                self.max_error = max(self.max_error, float(np.max(np.abs(error))))
                return pose.qpos.copy()
            blocks = []
            for geom_id in sim.foot_geom_ids:
                mujoco.mj_jacGeom(sim.model, pose, self.jacp, self.jacr, geom_id)
                blocks.append(self.jacp[:, self.variables].copy())
            mujoco.mj_jacSubtreeCom(sim.model, pose, self.jaccom, sim.trunk_id)
            blocks.append(self.jaccom[:, self.variables].copy())
            jacobian = np.vstack(blocks)
            delta = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + 1e-8 * np.eye(15), error
            )
            delta *= min(1.0, 0.05 / max(float(np.max(np.abs(delta))), 1e-12))
            velocity = np.zeros(sim.model.nv)
            velocity[self.variables] = delta
            mujoco.mj_integratePos(sim.model, pose.qpos, velocity, 1.0)
            pose.qpos[sim.q_indices] = np.clip(
                pose.qpos[sim.q_indices], self.limits[:, 0], self.limits[:, 1]
            )
        raise RuntimeError("Whole-body IK did not converge")


class FrontRightStepSimulation(WeightShiftSimulation):
    def __init__(self, model_path):
        super().__init__(model_path)
        self.step_length = 0.05
        self.lift_height = 0.05
        self.unload_start, self.unload_end = 7.0, 8.0
        self.lift_start, self.lift_end = 8.0, 9.0
        self.advance_end = 10.5
        self.lower_end = 11.5
        self.reload_start, self.reload_end = 12.0, 13.0
        self.recenter_start, self.recenter_end = 14.0, 17.0
        self.duration = 19.0
        self.horizon = 15
        self.mpc_dt = self.steps_per_mpc * self.model.opt.timestep
        self.fr_stance_limit = float(self.forces[0, 2])
        self.landed_foot_goals = self.foot_goals.copy()
        self.landed_foot_goals[0, 0] += self.step_length
        self.final_com_goal = self.nominal_com.copy()
        self.final_com_goal[0] += self.step_length / 4.0
        self.swing_pose_plan, self.recenter_pose_plan = self._plan_step()
        self.swing_plan = self.swing_pose_plan[:, self.q_indices]
        self.recenter_plan = self.recenter_pose_plan[:, self.q_indices]
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

    def planned_feet(self, seconds):
        goals = self.foot_goals.copy()
        lift = smoothstep((seconds - self.lift_start) / (self.lift_end - self.lift_start))
        lower = smoothstep((seconds - self.advance_end) / (self.lower_end - self.advance_end))
        advance = smoothstep((seconds - self.lift_end) / (self.advance_end - self.lift_end))
        goals[0, 0] += self.step_length * advance
        goals[0, 2] += self.lift_height * (lift - lower)
        return goals

    def _plan_step(self):
        pose = mujoco.MjData(self.model)
        pose.qpos[:] = self.data.qpos
        pose.qpos[self.q_indices] = self.posture_plan[-1]
        mujoco.mj_forward(self.model, pose)
        base_joint = self.model.body_jntadr[self.trunk_id]
        base_qpos = self.model.jnt_qposadr[base_joint]
        pose.qpos[base_qpos:base_qpos + 3] += (
            self.foot_goals[1] - pose.geom_xpos[self.foot_geom_ids[1]]
        )
        ik = WholeBodyIK(self, pose.qpos)
        samples = round((self.lower_end - self.lift_start) / self.mpc_dt) + 1
        swing = [
            ik.solve(self.planned_feet(seconds), self.target_com)
            for seconds in np.linspace(self.lift_start, self.lower_end, samples)
        ]
        recenter = [
            ik.solve(
                self.landed_foot_goals,
                self.target_com + fraction * (self.final_com_goal - self.target_com),
            )
            for fraction in np.linspace(0.0, 1.0, 51)
        ]
        self.ik_max_error = max(self.ik_max_error, ik.max_error)
        return np.array(swing), np.array(recenter)

    def _update_goals(self):
        model, data = self.model, self.data
        mujoco.mj_forward(model, data)
        seconds = self.step_count * model.opt.timestep
        self.alpha = smoothstep((seconds - self.shift_start) / self.shift_duration)
        self.q_target = interpolate(self.posture_plan, self.alpha)
        self.x_ref[3:6] = self.nominal_com + self.alpha * (self.target_com - self.nominal_com)
        if seconds >= self.lift_start:
            fraction = (seconds - self.lift_start) / (self.lower_end - self.lift_start)
            self.q_target = interpolate(self.swing_plan, fraction)
        if seconds >= self.recenter_start:
            fraction = smoothstep(
                (seconds - self.recenter_start) / (self.recenter_end - self.recenter_start)
            )
            self.q_target = interpolate(self.recenter_plan, fraction)
            self.x_ref[3:6] = self.target_com + fraction * (self.final_com_goal - self.target_com)
        phase_ends = [
            (self.shift_start, "settle"),
            (self.shift_start + self.shift_duration, "shift COM"),
            (self.unload_start, "ready"),
            (self.unload_end, "unload FR"),
            (self.lift_end, "lift FR"),
            (self.advance_end, "advance FR"),
            (self.lower_end, "lower FR"),
            (self.reload_start, "touchdown"),
            (self.reload_end, "reload FR"),
            (self.recenter_start, "four feet"),
            (self.recenter_end, "recenter COM"),
            (np.inf, "finish"),
        ]
        self.phase_name = next(name for end, name in phase_ends if seconds < end)

        mass = model.body_subtreemass[self.trunk_id]
        com = data.subtree_com[self.trunk_id].copy()
        inertia = robot_inertia(model, data, self.trunk_id, com)
        contacts = data.geom_xpos[self.foot_geom_ids].copy()
        contacts[:, 2] -= model.geom_size[self.foot_geom_ids, 0]
        lever = contacts - com
        balance = np.vstack((np.ones(4), lever[:, 1], -lever[:, 0]))
        target = np.array([-mass * model.opt.gravity[2], 0.0, 0.0])
        fz = np.linalg.lstsq(balance, target, rcond=None)[0]
        fz[0] = max(0.0, fz[0]) * self.availability(seconds)
        fz[1:] = np.linalg.solve(balance[:, 1:], target - balance[:, 0] * fz[0])
        if np.min(fz) < -1e-6:
            raise RuntimeError("COM is outside the active support polygon")
        force_ref = np.zeros((4, 3))
        force_ref[:, 2] = np.maximum(fz, 0.0)
        self.contact_limits = np.full((self.horizon, 4), 100.0)
        for step in range(self.horizon):
            future = seconds + step * self.mpc_dt
            if future >= self.unload_start:
                self.contact_limits[step, 0] = self.fr_stance_limit * self.availability(future)
        Ad, Bd, cd = build_mpc_model(mass, inertia, contacts, com, model.opt.gravity, dt=self.mpc_dt)
        self.mpc = ContactMPC(
            Ad, Bd, cd, self.x_ref, force_ref.ravel(), self.contact_limits,
            horizon=self.horizon,
        )

    def fr_clearance(self):
        geom_id = self.foot_geom_ids[0]
        floor_id = self.model.geom("floor").id
        return float(
            self.data.geom_xpos[geom_id, 2] - self.model.geom_size[geom_id, 0]
            - self.data.geom_xpos[floor_id, 2]
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()
    project = Path(__file__).parent
    sim = FrontRightStepSimulation(project / "models/unitree_a1/scene_mpc.xml")
    csv_path = args.csv or project / "logs" / f"step_fr_{datetime.now():%Y%m%d_%H%M%S_%f}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    print("FR step: lift 8-9 s, advance 9-10.5 s, lower 10.5-11.5 s.")
    print("Reload 12-13 s, recenter COM 14-17 s. Total: 19 s.")
    print(f"Step length: {sim.step_length * 1000:.0f} mm; lift: {sim.lift_height * 1000:.0f} mm")
    print(f"IK maximum error: {sim.ik_max_error:.2e} m")
    print(f"CSV: {csv_path}")
    context = nullcontext(None) if args.headless else mujoco.viewer.launch_passive(sim.model, sim.data)
    total_steps = round(sim.duration / sim.model.opt.timestep)
    peak_torque = peak_tilt = max_drift = max_swing_force = 0.0
    clipped_steps = transit_samples = 0
    min_transit_clearance = min_transit_margin = min_transit_load = np.inf
    max_transit_fr_load = 0.0
    pre_step_fr = None
    with context as viewer, csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "time_s", "phase", "mpc_status", "com_x", "com_y", "com_z",
            "roll_rad", "pitch_rad", "fr_x", "fr_y", "fr_clearance_m",
            "fr_load_n", "fl_load_n", "rr_load_n", "rl_load_n",
            "triangle_margin_m", "support_drift_m", "inactive_fr_force_n",
            "max_torque_nm", "qp_s",
        ])
        while sim.step_count < total_steps:
            if viewer is not None and not viewer.is_running():
                break
            started = time.perf_counter()
            sim.step()
            torque = float(np.max(np.abs(sim.unclipped_torque)))
            peak_torque = max(peak_torque, torque)
            limits = sim.model.actuator_ctrlrange
            clipped_steps += int(np.any(
                (sim.unclipped_torque < limits[:, 0] - 1e-8)
                | (sim.unclipped_torque > limits[:, 1] + 1e-8)
            ))
            if not np.isfinite(sim.data.qpos).all() or sim.data.qpos[2] < 0.15:
                raise RuntimeError("Step failed: non-finite state or low trunk")
            if sim.step_count % sim.steps_per_mpc == 1:
                mujoco.mj_forward(sim.model, sim.data)
                state = sim.read_state()
                loads = foot_ground_loads(sim)
                clearance = sim.fr_clearance()
                margin = sim.future_support_margin()
                feet = sim.data.geom_xpos[sim.foot_geom_ids].copy()
                seconds = (sim.step_count - 1) * sim.model.opt.timestep
                peak_tilt = max(peak_tilt, float(np.max(np.abs(state[:2]))))
                drift = float(np.max(np.linalg.norm(feet[1:, :2] - sim.foot_goals[1:, :2], axis=1)))
                max_drift = max(max_drift, drift)
                if seconds < sim.lift_start:
                    pre_step_fr = feet[0].copy()
                inactive_force = 0.0
                if sim.unload_end <= seconds < sim.reload_start:
                    inactive_force = float(np.max(np.abs(sim.forces[0])))
                    max_swing_force = max(max_swing_force, inactive_force)
                if sim.lift_end + 0.2 <= seconds < sim.advance_end:
                    transit_samples += 1
                    min_transit_clearance = min(min_transit_clearance, clearance)
                    min_transit_margin = min(min_transit_margin, margin)
                    min_transit_load = min(min_transit_load, float(np.min(loads[1:])))
                    max_transit_fr_load = max(max_transit_fr_load, abs(float(loads[0])))
                writer.writerow([
                    sim.data.time, sim.phase_name, sim.status, *state[3:6], *state[:2],
                    *feet[0, :2], clearance, *loads, margin, drift, inactive_force,
                    torque, sim.mpc.last_info.run_time,
                ])
                if (sim.step_count - 1) % round(1.0 / sim.model.opt.timestep) == 0:
                    print(
                        f"t={seconds:.1f}s | {sim.phase_name} | MPC={sim.status} | "
                        f"FR x={feet[0, 0]:.4f}m | clearance={clearance * 1000:.1f}mm | "
                        f"load={loads[0]:.2f}N"
                    )
            if viewer is not None:
                viewer.sync()
                remaining = sim.model.opt.timestep - (time.perf_counter() - started)
                if remaining > 0:
                    time.sleep(remaining)

    mujoco.mj_forward(sim.model, sim.data)
    final = sim.read_state()
    feet = sim.data.geom_xpos[sim.foot_geom_ids].copy()
    loads = foot_ground_loads(sim)
    complete = sim.step_count == total_steps
    landing_error = float(np.linalg.norm(feet[0, :2] - sim.landed_foot_goals[0, :2]))
    com_error = float(np.linalg.norm(final[3:5] - sim.final_com_goal[:2]))
    advancement = float(feet[0, 0] - pre_step_fr[0]) if pre_step_fr is not None else np.nan
    print("COMPLETE" if complete else "STOPPED EARLY")
    print(f"Actual FR advance from pre-step stance: {advancement * 1000:.2f} mm")
    print(f"Final FR XY target error: {landing_error * 1000:.2f} mm")
    if transit_samples:
        print(f"Transfer minimum FR clearance: {min_transit_clearance * 1000:.2f} mm")
        print(f"Transfer maximum actual FR load: {max_transit_fr_load:.4f} N")
        print(f"Transfer minimum other foot load: {min_transit_load:.2f} N")
        print(f"Transfer minimum support margin: {min_transit_margin * 1000:.2f} mm")
    print(f"Maximum MPC force component on inactive FR: {max_swing_force:.2e} N")
    print(f"Maximum support foot XY drift: {max_drift * 1000:.2f} mm")
    print(f"Final COM XY target error: {com_error * 1000:.2f} mm")
    print(f"Peak roll/pitch: {np.degrees(peak_tilt):.3f} deg")
    print(f"Maximum commanded torque: {peak_torque:.3f} Nm; clipped steps: {clipped_steps}")
    print("Final actual ground loads FR/FL/RR/RL, N:", np.round(loads, 2))
    ready = (
        complete and transit_samples > 0 and abs(advancement - sim.step_length) < 0.01
        and landing_error < 0.01 and min_transit_clearance > 0.025
        and max_transit_fr_load < 0.1 and min_transit_load > 5.0
        and min_transit_margin > 0.03 and max_swing_force < 1e-3
        and np.degrees(peak_tilt) < 3.0 and max_drift < 0.01
        and com_error < 0.01 and clipped_steps == 0 and np.all(loads > 1.0)
    )
    print("FR STEP PASSED" if ready else "CHECK FR STEP BEFORE CONTINUING")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()

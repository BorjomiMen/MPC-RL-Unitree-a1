import argparse
from contextlib import nullcontext
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np

from contact_mpc import ContactMPC
from lift_fr import interpolate, smoothstep
from mpc_model import build_mpc_model
from stand_mpc import StandingSimulation
from step_fr import WholeBodyIK
from weight_shift import foot_ground_loads, robot_inertia, triangle_margin


LEG_NAMES = ("FR", "FL", "RR", "RL")


def support_com_target(desired, triangle, margin=0.04):
    """Project desired XY onto a triangle with edges inset by margin."""
    following = np.roll(triangle, -1, axis=0)
    edges = following - triangle
    area_twice = np.sum(triangle[:, 0] * following[:, 1] - triangle[:, 1] * following[:, 0])
    lengths = np.linalg.norm(edges, axis=1)
    if abs(area_twice) < 1e-10 or np.min(lengths) < 1e-8:
        raise ValueError("Degenerate support triangle")
    normals = np.sign(area_twice) * np.c_[-edges[:, 1], edges[:, 0]] / lengths[:, None]
    bounds = np.sum(normals * triangle, axis=1) + margin
    candidates = [np.asarray(desired, dtype=float)]
    for i in range(3):
        candidates.append(desired + (bounds[i] - normals[i] @ desired) * normals[i])
        for j in range(i):
            matrix = normals[[i, j]]
            if abs(np.linalg.det(matrix)) > 1e-10:
                candidates.append(np.linalg.solve(matrix, bounds[[i, j]]))
    feasible = [point for point in candidates if np.all(normals @ point >= bounds - 1e-10)]
    if not feasible:
        raise ValueError("Requested support margin leaves no feasible COM target")
    return min(feasible, key=lambda point: np.sum((point - desired)**2)).copy()


@dataclass
class CrawlStep:
    leg: int
    start: float
    feet_before: np.ndarray
    com_before: np.ndarray
    com_support: np.ndarray

    @property
    def unload_start(self):
        return self.start + 3.4

    @property
    def lift_start(self):
        return self.start + 4.2

    @property
    def lift_end(self):
        return self.start + 5.0

    @property
    def advance_end(self):
        return self.start + 6.2

    @property
    def lower_end(self):
        return self.start + 7.0

    @property
    def reload_start(self):
        return self.start + 7.4

    @property
    def reload_end(self):
        return self.start + 8.2

    @property
    def end(self):
        return self.start + 8.6


class CrawlSimulation(StandingSimulation):
    def __init__(self, model_path):
        super().__init__(model_path)
        self.step_length, self.lift_height = 0.05, 0.05
        self.settle_end = 3.0
        self.nominal_com = self.x_ref[3:6].copy()
        self.initial_feet = self.data.geom_xpos[self.foot_geom_ids].copy()
        self.horizon = 15
        self.mpc_dt = self.steps_per_mpc * self.model.opt.timestep
        self.steps = []
        feet, previous_com, start = self.initial_feet.copy(), self.nominal_com.copy(), self.settle_end
        for leg in (0, 3, 1, 2):
            support = [index for index in range(4) if index != leg]
            neutral = self.nominal_com.copy()
            neutral[:2] += np.mean(feet[:, :2] - self.initial_feet[:, :2], axis=0)
            com_goal = neutral.copy()
            com_goal[:2] = support_com_target(neutral[:2], feet[support, :2])
            step = CrawlStep(leg, start, feet.copy(), previous_com.copy(), com_goal)
            self.steps.append(step)
            feet[leg, 0] += self.step_length
            previous_com, start = com_goal.copy(), step.end
        self.final_feet = feet.copy()
        self.final_com_goal = self.nominal_com.copy()
        self.final_com_goal[0] += self.step_length
        self.recenter_start = self.steps[-1].end
        self.recenter_end = self.recenter_start + 3.0
        self.duration = round(self.recenter_end + 2.0, 6)
        self.plan_times = np.linspace(0.0, self.duration, round(self.duration / self.mpc_dt) + 1)
        ik = WholeBodyIK(self, self.data.qpos)
        poses = []
        for seconds in self.plan_times:
            goals, com_goal, step, phase = self.configuration(seconds)
            try:
                poses.append(ik.solve(goals, com_goal))
            except RuntimeError as error:
                raise RuntimeError(f"Crawl IK failed at {seconds:.2f}s, {phase}") from error
        self.pose_plan = np.array(poses)
        self.joint_plan = self.pose_plan[:, self.q_indices]
        self.ik_max_error = ik.max_error
        self.active_step = None
        self.phase_name = "settle"
        self.foot_goals = self.initial_feet.copy()

    def configuration(self, seconds):
        if seconds < self.settle_end:
            return self.initial_feet.copy(), self.nominal_com.copy(), None, "settle"
        for step in self.steps:
            if seconds < step.end:
                feet = step.feet_before.copy()
                com = step.com_before + smoothstep((seconds - step.start) / 3.0) * (
                    step.com_support - step.com_before
                )
                lift = smoothstep((seconds - step.lift_start) / (step.lift_end - step.lift_start))
                lower = smoothstep((seconds - step.advance_end) / (step.lower_end - step.advance_end))
                advance = smoothstep((seconds - step.lift_end) / (step.advance_end - step.lift_end))
                feet[step.leg, 0] += self.step_length * advance
                feet[step.leg, 2] += self.lift_height * (lift - lower)
                phase_ends = [
                    (step.start + 3.0, "shift COM"),
                    (step.unload_start, "ready"), (step.lift_start, "unload"),
                    (step.lift_end, "lift"), (step.advance_end, "advance"),
                    (step.lower_end, "lower"), (step.reload_start, "touchdown"),
                    (step.reload_end, "reload"), (step.end, "four feet"),
                ]
                phase = next(name for end, name in phase_ends if seconds < end)
                return feet, com, step, f"{LEG_NAMES[step.leg]} {phase}"
        fraction = smoothstep((seconds - self.recenter_start) / 3.0)
        com = self.steps[-1].com_support + fraction * (self.final_com_goal - self.steps[-1].com_support)
        return self.final_feet.copy(), com, None, "recenter COM" if seconds < self.recenter_end else "finish"

    @staticmethod
    def availability(step, seconds):
        if seconds < step.unload_start:
            return 1.0
        if seconds < step.lift_start:
            return float(np.clip(
                1.0 - smoothstep((seconds - step.unload_start) / (step.lift_start - step.unload_start)),
                0.0, 1.0,
            ))
        if seconds < step.reload_start:
            return 0.0
        return float(np.clip(
            smoothstep((seconds - step.reload_start) / (step.reload_end - step.reload_start)),
            0.0, 1.0,
        ))

    def _update_goals(self):
        model, data = self.model, self.data
        mujoco.mj_forward(model, data)
        seconds = self.step_count * model.opt.timestep
        self.foot_goals, com_goal, self.active_step, self.phase_name = self.configuration(seconds)
        self.q_target = interpolate(self.joint_plan, seconds / self.duration)
        self.x_ref[3:6] = com_goal
        mass = model.body_subtreemass[self.trunk_id]
        com = data.subtree_com[self.trunk_id].copy()
        inertia = robot_inertia(model, data, self.trunk_id, com)
        contacts = data.geom_xpos[self.foot_geom_ids].copy()
        contacts[:, 2] -= model.geom_size[self.foot_geom_ids, 0]
        lever = contacts - com
        balance = np.vstack((np.ones(4), lever[:, 1], -lever[:, 0]))
        target = np.array([-mass * model.opt.gravity[2], 0.0, 0.0])
        fz = np.linalg.lstsq(balance, target, rcond=None)[0]
        if self.active_step is not None:
            leg = self.active_step.leg
            support = [index for index in range(4) if index != leg]
            fz[leg] = max(0.0, fz[leg]) * self.availability(self.active_step, seconds)
            fz[support] = np.linalg.solve(balance[:, support], target - balance[:, leg] * fz[leg])
        if np.min(fz) < -1e-6:
            raise RuntimeError(f"COM outside active support polygon at {seconds:.2f}s")
        force_ref = np.zeros((4, 3))
        force_ref[:, 2] = np.maximum(fz, 0.0)
        self.contact_limits = np.full((self.horizon, 4), 100.0)
        for index in range(self.horizon):
            future = seconds + index * self.mpc_dt
            for step in self.steps:
                if step.unload_start <= future < step.end:
                    self.contact_limits[index, step.leg] = 40.0 * self.availability(step, future)
                    break
        Ad, Bd, cd = build_mpc_model(mass, inertia, contacts, com, model.opt.gravity, dt=self.mpc_dt)
        self.mpc = ContactMPC(
            Ad, Bd, cd, self.x_ref, force_ref.ravel(), self.contact_limits, horizon=self.horizon,
        )

    def step(self):
        if self.step_count % self.steps_per_mpc == 0:
            self._update_goals()
        super().step()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()
    project = Path(__file__).parent
    print("Planning one crawl cycle: FR -> RL -> FL -> RR ...", flush=True)
    sim = CrawlSimulation(project / "models/unitree_a1/scene_mpc.xml")
    csv_path = args.csv or project / "logs" / f"crawl_{datetime.now():%Y%m%d_%H%M%S_%f}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Step/lift: 50/50 mm; total: {sim.duration:.1f} s; IK maximum error: {sim.ik_max_error:.2e} m")
    for step in sim.steps:
        print(f"{LEG_NAMES[step.leg]}: lift {step.lift_start:.1f}, advance {step.lift_end:.1f}, lower {step.advance_end:.1f}, reload {step.reload_start:.1f} s")
    print(f"CSV: {csv_path}")
    metrics = {
        step.leg: dict(samples=0, clearance=np.inf, margin=np.inf, support_load=np.inf,
                       swing_load=0.0, pre_foot=None, advance=np.nan, landing_error=np.nan)
        for step in sim.steps
    }
    context = nullcontext(None) if args.headless else mujoco.viewer.launch_passive(sim.model, sim.data)
    peak_torque = peak_tilt = max_drift = max_inactive_force = 0.0
    clipped_steps = 0
    baseline_com = None
    total_steps = round(sim.duration / sim.model.opt.timestep)
    with context as viewer, csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "time_s", "phase", "mpc_status", "com_x", "com_y", "com_z", "roll_rad", "pitch_rad",
            *[f"{name.lower()}_{axis}" for name in LEG_NAMES for axis in ("x", "y", "z")],
            *[f"{name.lower()}_load_n" for name in LEG_NAMES], "active_leg", "swing_clearance_m",
            "triangle_margin_m", "support_drift_m", "inactive_force_n", "max_torque_nm", "clipped", "qp_s",
        ])
        while sim.step_count < total_steps:
            if viewer is not None and not viewer.is_running():
                break
            started = time.perf_counter()
            sim.step()
            torque = float(np.max(np.abs(sim.unclipped_torque)))
            peak_torque = max(peak_torque, torque)
            limits = sim.model.actuator_ctrlrange
            clipped = bool(np.any((sim.unclipped_torque < limits[:, 0] - 1e-8) | (sim.unclipped_torque > limits[:, 1] + 1e-8)))
            clipped_steps += int(clipped)
            if not np.isfinite(sim.data.qpos).all() or sim.data.qpos[2] < 0.15:
                raise RuntimeError("Crawl failed: non-finite state or low trunk")
            if sim.step_count % sim.steps_per_mpc == 1:
                mujoco.mj_forward(sim.model, sim.data)
                state = sim.read_state()
                feet = sim.data.geom_xpos[sim.foot_geom_ids].copy()
                loads = foot_ground_loads(sim)
                seconds = (sim.step_count - 1) * sim.model.opt.timestep
                peak_tilt = max(peak_tilt, float(np.max(np.abs(state[:2]))))
                if seconds < sim.settle_end:
                    baseline_com = state[3:6].copy()
                active = sim.active_step
                stance = np.ones(4, dtype=bool)
                clearance = margin = np.nan
                inactive_force = 0.0
                if active is not None:
                    leg = active.leg
                    info = metrics[leg]
                    if seconds < active.lift_start:
                        info["pre_foot"] = feet[leg].copy()
                    if active.lift_start <= seconds < active.reload_start:
                        stance[leg] = False
                        inactive_force = float(np.max(np.abs(sim.forces[leg])))
                        max_inactive_force = max(max_inactive_force, inactive_force)
                    geom_id = sim.foot_geom_ids[leg]
                    floor_height = sim.data.geom_xpos[sim.model.geom("floor").id, 2]
                    clearance = feet[leg, 2] - sim.model.geom_size[geom_id, 0] - floor_height
                    support = [index for index in range(4) if index != leg]
                    margin = triangle_margin(state[3:5], feet[support, :2])
                    if active.lift_end + 0.15 <= seconds < active.advance_end:
                        info["samples"] += 1
                        info["clearance"] = min(info["clearance"], clearance)
                        info["margin"] = min(info["margin"], margin)
                        info["support_load"] = min(info["support_load"], float(np.min(loads[support])))
                        info["swing_load"] = max(info["swing_load"], abs(float(loads[leg])))
                    if seconds >= active.reload_end and np.isnan(info["advance"]):
                        info["advance"] = float(feet[leg, 0] - info["pre_foot"][0])
                        target_xy = active.feet_before[leg, :2] + [sim.step_length, 0.0]
                        info["landing_error"] = float(np.linalg.norm(feet[leg, :2] - target_xy))
                drift = float(np.max(np.linalg.norm(feet[stance, :2] - sim.foot_goals[stance, :2], axis=1)))
                max_drift = max(max_drift, drift)
                writer.writerow([
                    sim.data.time, sim.phase_name, sim.status, *state[3:6], *state[:2], *feet.ravel(), *loads,
                    LEG_NAMES[active.leg] if active is not None else "", clearance, margin, drift,
                    inactive_force, torque, int(clipped), sim.mpc.last_info.run_time,
                ])
                if (sim.step_count - 1) % round(2.0 / sim.model.opt.timestep) == 0:
                    print(f"t={seconds:.1f}s | {sim.phase_name} | MPC={sim.status} | COM xy={np.round(state[3:5], 4)}")
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
    com_advance = final[3] - baseline_com[0] if baseline_com is not None else np.nan
    com_error = float(np.linalg.norm(final[3:5] - sim.final_com_goal[:2]))
    final_foot_error = float(np.max(np.linalg.norm(feet[:, :2] - sim.final_feet[:, :2], axis=1)))
    print("COMPLETE" if complete else "STOPPED EARLY")
    legs_passed = True
    for step in sim.steps:
        info = metrics[step.leg]
        passed = (
            info["samples"] > 0 and abs(info["advance"] - sim.step_length) < 0.01
            and info["landing_error"] < 0.01 and info["clearance"] > 0.025
            and info["margin"] > 0.025 and info["support_load"] > 5.0 and info["swing_load"] < 0.1
        )
        legs_passed &= passed
        print(
            f"{LEG_NAMES[step.leg]}: advance={info['advance'] * 1000:.2f}mm | "
            f"landing error={info['landing_error'] * 1000:.2f}mm | "
            f"transfer clearance>={info['clearance'] * 1000:.2f}mm | "
            f"margin>={info['margin'] * 1000:.2f}mm | swing load<={info['swing_load']:.4f}N | "
            f"{'PASSED' if passed else 'CHECK'}"
        )
    print(f"COM forward advance: {com_advance * 1000:.2f} mm")
    print(f"Final COM XY target error: {com_error * 1000:.2f} mm")
    print(f"Final maximum foot XY target error: {final_foot_error * 1000:.2f} mm")
    print(f"Maximum stance foot XY target error: {max_drift * 1000:.2f} mm")
    print(f"Maximum MPC force component on inactive foot: {max_inactive_force:.2e} N")
    print(f"Peak roll/pitch: {np.degrees(peak_tilt):.3f} deg")
    print(f"Maximum commanded torque: {peak_torque:.3f} Nm; clipped steps: {clipped_steps}")
    print("Final actual ground loads FR/FL/RR/RL, N:", np.round(loads, 2))
    ready = (
        complete and legs_passed and abs(com_advance - sim.step_length) < 0.01
        and com_error < 0.01 and final_foot_error < 0.01 and max_drift < 0.01
        and max_inactive_force < 1e-3 and np.degrees(peak_tilt) < 3.0
        and clipped_steps == 0 and np.all(loads > 5.0) and np.linalg.norm(final[9:11]) < 0.01
    )
    print("CRAWL CYCLE PASSED" if ready else "CHECK CRAWL BEFORE CONTINUING")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()

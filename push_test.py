import argparse
from contextlib import nullcontext
import csv
from datetime import datetime
from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np

from stand_mpc import StandingSimulation


class FixedForces:
    def __init__(self, forces):
        self.forces = forces.copy()
        self.last_info = None

    def solve(self, state):
        return self.forces.copy(), "fixed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--mode", choices=("mpc", "fixed"), default="mpc")
    parser.add_argument("--force", type=float, default=50.0)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()
    if not np.isfinite(args.force) or args.force <= 0:
        parser.error("--force must be a finite positive magnitude in newtons")

    project_dir = Path(__file__).parent
    sim = StandingSimulation(project_dir / "models/unitree_a1/scene_mpc.xml")
    if args.mode == "fixed":
        sim.mpc = FixedForces(sim.forces)

    dt = sim.model.opt.timestep
    push_start = round(5.0 / dt)
    push_end = push_start + round(0.2 / dt)
    total_steps = round(10.0 / dt)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    csv_path = args.csv or project_dir / "logs" / f"push_{args.mode}_{stamp}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    context = (
        nullcontext(None) if args.headless
        else mujoco.viewer.launch_passive(sim.model, sim.data)
    )

    before_push = None
    peak_x_shift = 0.0
    peak_tilt = 0.0
    clipped_steps = 0
    peak_torque = 0.0
    print(f"Mode: {args.mode}; push: {args.force:g} N from 5.0 to 5.2 s.")
    print(f"CSV: {csv_path}")

    with context as viewer, csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "t_s", "push_fx_N", "com_x_m", "com_y_m", "com_z_m",
            "roll_rad", "pitch_rad", "yaw_rad", "vx_m_s", "vy_m_s", "vz_m_s",
            "cmd_fx_total_N", "cmd_fy_total_N", "cmd_fz_total_N",
            "max_abs_tau_cmd_Nm", "torque_clipped", "status", "qp_ms",
        ])
        while sim.step_count < total_steps:
            if viewer is not None and not viewer.is_running():
                break
            wall_start = time.perf_counter()
            step = sim.step_count
            if step == push_start:
                mujoco.mj_forward(sim.model, sim.data)
                before_push = sim.read_state().copy()
                print("PUSH ON")
            if step == push_end:
                print("PUSH OFF")

            pushing = push_start <= step < push_end
            sim.data.xfrc_applied[sim.trunk_id, :] = 0.0
            sim.data.xfrc_applied[sim.trunk_id, 0] = args.force if pushing else 0.0
            sim.step()

            limits = sim.model.actuator_ctrlrange
            clipped = bool(np.any(
                (sim.unclipped_torque < limits[:, 0] - 1e-8)
                | (sim.unclipped_torque > limits[:, 1] + 1e-8)
            ))
            clipped_steps += int(clipped)
            torque_max = float(np.max(np.abs(sim.unclipped_torque)))
            peak_torque = max(peak_torque, torque_max)

            if step % sim.steps_per_mpc == 0:
                x = sim.state
                if before_push is not None:
                    peak_x_shift = max(peak_x_shift, abs(float(x[3] - before_push[3])))
                    peak_tilt = max(peak_tilt, float(np.max(np.abs(x[:2]))))
                info = sim.mpc.last_info
                qp_ms = float(info.run_time * 1000) if info is not None else float("nan")
                writer.writerow([
                    step * dt, args.force if pushing else 0.0,
                    *x[3:6], *x[:3], *x[9:12], *sim.forces.sum(axis=0),
                    torque_max, int(clipped), sim.status, qp_ms,
                ])
            if step % round(1.0 / dt) == 0:
                print(
                    f"t={step * dt:.1f}s | status={sim.status} | "
                    f"x={sim.state[3]:+.4f}m | z={sim.state[5]:.4f}m"
                )
            if not np.isfinite(sim.data.qpos).all() or sim.data.qpos[2] < 0.15:
                raise RuntimeError("Standing test failed: non-finite state or low trunk")
            if viewer is not None:
                viewer.sync()
                remaining = dt - (time.perf_counter() - wall_start)
                if remaining > 0:
                    time.sleep(remaining)

    sim.data.xfrc_applied[sim.trunk_id, :] = 0.0
    mujoco.mj_forward(sim.model, sim.data)
    final = sim.read_state()
    complete = sim.step_count == total_steps
    print("COMPLETE" if complete else "STOPPED EARLY")
    print(f"Max commanded torque: {peak_torque:.3f} Nm; clipped steps: {clipped_steps}")
    if before_push is not None:
        print(f"Peak COM x shift after push: {peak_x_shift * 1000:.3f} mm")
        print(f"Peak |roll/pitch| after push: {np.degrees(peak_tilt):.3f} deg")
        print(f"Final COM x shift from pre-push stance: {(final[3] - before_push[3]) * 1000:.3f} mm")
        print(f"Final COM vx: {final[9]:+.6f} m/s")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()

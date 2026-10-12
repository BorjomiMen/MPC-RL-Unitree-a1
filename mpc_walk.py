import argparse
from contextlib import nullcontext
from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np
import osqp
from scipy import sparse

from crawl_test import CrawlStep, LEG_NAMES, support_com_target
from lift_fr import interpolate, smoothstep
from mpc_model import build_mpc_model
from stand_mpc import StandingSimulation
from step_fr import WholeBodyIK
from weight_shift import foot_ground_loads, robot_inertia, triangle_margin


class GaitPlan:
    def __init__(self, feet, com, vx=0.001, vy=0.0, cycles=2):
        velocity = np.array([vx, vy], dtype=float)

        if (
            not np.isfinite(velocity).all()
            or np.linalg.norm(velocity) > 0.05 / 34.4 + 1e-12
        ):
            raise ValueError("Average speed must be <= 0.05 / 34.4 m/s")

        if not isinstance(cycles, int) or not 1 <= cycles <= 10:
            raise ValueError("cycles must be from 1 to 10")

        self.feet0, self.com0 = feet.copy(), com.copy()
        self.stride = velocity * 34.4
        self.steps = []

        feet = feet.copy()
        previous = com.copy()
        start = 3.0
        moving = np.linalg.norm(velocity) > 1e-12

        for _ in range(cycles if moving else 0):
            for leg in (0, 3, 1, 2):
                support = [i for i in range(4) if i != leg]

                goal = com.copy()
                goal[:2] += np.mean(
                    feet[:, :2] - self.feet0[:, :2], axis=0
                )
                goal[:2] = support_com_target(
                    goal[:2], feet[support, :2], margin=0.04
                )

                step = CrawlStep(
                    leg, start, feet.copy(), previous.copy(), goal
                )
                self.steps.append(step)

                feet[leg, :2] += self.stride
                previous, start = goal.copy(), step.end

        self.final_feet = feet.copy()
        self.final_com = com.copy()
        self.final_com[:2] += cycles * self.stride if moving else 0.0

        self.recenter_start = start
        self.duration = round(start + 5.0, 6)

        self.gates = [
            (t, kind, s)
            for s in self.steps
            for t, kind in (
                (s.lift_start, "unload"),
                (s.lift_end, "clear"),
                (s.reload_start, "land"),
            )
        ]

    @staticmethod
    def availability(s, t):
        if t < s.unload_start:
            return 1.0

        if t < s.lift_start:
            value = 1.0 - smoothstep(
                (t - s.unload_start)
                / (s.lift_start - s.unload_start)
            )
        elif t < s.reload_start:
            value = 0.0
        else:
            value = smoothstep(
                (t - s.reload_start)
                / (s.reload_end - s.reload_start)
            )

        return float(np.clip(value, 0.0, 1.0))

    @staticmethod
    def derivative(t, start, duration):
        u = (t - start) / duration

        if 0.0 < u < 1.0:
            return 30.0 * u**2 * (1.0 - u)**2 / duration

        return 0.0

    def sample(self, t):
        t = float(np.clip(t, 0.0, self.duration))

        if t < 3.0 or not self.steps:
            return (
                self.feet0.copy(),
                self.com0.copy(),
                np.zeros(3),
                None,
                "stand",
            )

        for s in self.steps:
            if t < s.end:
                feet = s.feet_before.copy()
                delta = s.com_support - s.com_before

                com = s.com_before + smoothstep(
                    (t - s.start) / 3.0
                ) * delta
                velocity = self.derivative(
                    t, s.start, 3.0
                ) * delta

                lift = smoothstep(
                    (t - s.lift_start)
                    / (s.lift_end - s.lift_start)
                )
                advance = smoothstep(
                    (t - s.lift_end)
                    / (s.advance_end - s.lift_end)
                )
                lower = smoothstep(
                    (t - s.advance_end)
                    / (s.lower_end - s.advance_end)
                )

                feet[s.leg, :2] += advance * self.stride
                feet[s.leg, 2] += 0.05 * (lift - lower)

                return feet, com, velocity, s, LEG_NAMES[s.leg]

        delta = self.final_com - self.steps[-1].com_support
        com = self.steps[-1].com_support + smoothstep(
            (t - self.recenter_start) / 3.0
        ) * delta
        velocity = self.derivative(
            t, self.recenter_start, 3.0
        ) * delta

        return self.final_feet.copy(), com, velocity, None, "finish"

    def limits(self, t):
        caps = np.full(4, 100.0)

        for s in self.steps:
            if s.unload_start <= t < s.end:
                caps[s.leg] = 40.0 * self.availability(s, t)
                break

        return caps
class Trajectory:
    def __init__(self, sim, plan, dt=0.02):
        self.plan = plan
        self.times = np.linspace(
            0.0,
            plan.duration,
            round(plan.duration / dt) + 1,
        )

        ik = WholeBodyIK(sim, sim.data.qpos)

        self.joints = np.array([
            ik.solve(
                plan.sample(t)[0],
                plan.sample(t)[1],
            )[sim.q_indices]
            for t in self.times
        ])

        self.rates = np.gradient(self.joints, dt, axis=0)
        self.ik_error = ik.max_error

    def posture(self, t, frozen=False):
        fraction = np.clip(t / self.plan.duration, 0.0, 1.0)
        q = interpolate(self.joints, fraction)

        dq = (
            np.zeros(12)
            if frozen
            else interpolate(self.rates, fraction)
        )

        return q, dq

    def reference(self, t, frozen=False):
        _, com, velocity, _, _ = self.plan.sample(t)

        x = np.zeros(12)
        x[3:6] = com

        if not frozen:
            x[9:12] = velocity

        return x

    def preview(self, t, N, dt, gate=np.inf, frozen=False):
        times = (
            np.full(N + 1, t)
            if frozen
            else np.minimum(
                t + dt * np.arange(N + 1), gate
            )
        )

        refs = np.array([
            self.reference(s, frozen) for s in times
        ])
        caps = np.array([
            self.plan.limits(s) for s in times[:-1]
        ])

        forces = []

        for seconds in times[:-1]:
            feet, com, _, step, _ = self.plan.sample(seconds)
            r = feet - com

            balance = np.vstack((
                np.ones(4),
                r[:, 1],
                -r[:, 0],
            ))
            target = np.array([self.weight, 0.0, 0.0])

            fz = np.linalg.lstsq(
                balance, target, rcond=None
            )[0]

            if step is not None:
                leg = step.leg
                support = [i for i in range(4) if i != leg]

                fz[leg] *= self.plan.availability(step, seconds)
                fz[support] = np.linalg.solve(
                    balance[:, support],
                    target - balance[:, leg] * fz[leg],
                )

            if np.min(fz) < -1e-5:
                raise RuntimeError("Invalid static force reference")

            force = np.zeros((4, 3))
            force[:, 2] = np.maximum(fz, 0.0)
            forces.append(force.ravel())

        return refs, np.array(forces), caps
class PersistentMPC:
    def __init__(self, horizon=15, mu=0.5):
        self.N = int(horizon)
        self.offset = 12 * (self.N + 1)
        self.force_end = self.offset + 20 * self.N
        self.rows = self.force_end + 12 * self.N
        self.cols = self.offset + 12 * self.N

        self.weights = np.tile(
            [120, 120, 30, 20, 20, 250, 2, 2, 1, 5, 5, 15],
            (self.N + 1, 1),
        ).astype(float)
        self.weights[-1] *= 5.0
        self.R = 0.001

        rows, cols, values = [], [], []

        def block(row, col, value):
            rr, cc = np.indices(value.shape)
            slots = np.arange(
                len(values), len(values) + value.size
            )

            rows.extend((row + rr).ravel())
            cols.extend((col + cc).ravel())
            values.extend(value.ravel())

            return slots

        rows.extend(range(self.offset))
        cols.extend(range(self.offset))
        values.extend(np.ones(self.offset))

        ad, bd, td = [], [], []

        for k in range(self.N):
            ad.append(block(
                12 * (k + 1),
                12 * k,
                np.zeros((12, 12)),
            ))
            bd.append(block(
                12 * (k + 1),
                self.offset + 12 * k,
                np.zeros((12, 12)),
            ))
            td.append(block(
                self.force_end + 12 * k,
                self.offset + 12 * k,
                np.zeros((12, 12)),
            ))

        a = mu / np.sqrt(2.0)
        cone = np.array([
            [1, 0, -a],
            [-1, 0, -a],
            [0, 1, -a],
            [0, -1, -a],
            [0, 0, 1.0],
        ])

        for k in range(self.N):
            for foot in range(4):
                block(
                    self.offset + 20 * k + 5 * foot,
                    self.offset + 12 * k + 3 * foot,
                    cone,
                )

        rows, cols = np.array(rows), np.array(cols)
        order = np.lexsort((rows, cols))
        inverse = np.empty_like(order)
        inverse[order] = np.arange(len(order))

        self.ad_slots, self.bd_slots, self.torque_slots = (
            inverse[np.array(slots)]
            for slots in (ad, bd, td)
        )

        indptr = np.r_[
            0,
            np.cumsum(
                np.bincount(cols, minlength=self.cols)
            ),
        ]

        self.A = sparse.csc_matrix(
            (
                np.array(values)[order],
                rows[order],
                indptr,
            ),
            shape=(self.rows, self.cols),
        )

        self.lower = np.full(self.rows, -np.inf)
        self.upper = np.zeros(self.rows)

        self.lower[:self.offset] = 0.0

        self.fz_rows = np.arange(
            self.offset + 4, self.force_end, 5
        )
        self.lower[self.fz_rows] = 0.0
        self.upper[self.fz_rows] = 100.0

        self.lower[self.force_end:] = -33.5
        self.upper[self.force_end:] = 33.5

        P = sparse.diags(
            np.r_[
                2.0 * self.weights.ravel(),
                np.full(12 * self.N, 2.0 * self.R),
            ],
            format="csc",
        )

        self.solver = osqp.OSQP()
        self.solver.setup(
            P=P,
            q=np.zeros(self.cols),
            A=self.A,
            l=self.lower,
            u=self.upper,
            verbose=False,
            warm_starting=True,
            polishing=True,
            eps_abs=1e-6,
            eps_rel=1e-6,
            max_iter=10000,
        )

        self.last_info = None
        self.last_violation = 0.0
        self.wall_ms = 0.0

    @staticmethod
    def array(value, shape):
        result = np.broadcast_to(
            np.asarray(value, dtype=float), shape
        )

        if not np.isfinite(result).all():
            raise ValueError("Non-finite QP input")

        return result
    
    def solve(
        self, x, Ad, Bd, cd, refs, frefs,
        caps, T, bias, tau_min, tau_max,
    ):
        started = time.perf_counter()
        N = self.N

        x = self.array(x, (12,))
        Ad, Bd = (
            self.array(a, (N, 12, 12))
            for a in (Ad, Bd)
        )
        cd = self.array(cd, (N, 12))
        refs = self.array(refs, (N + 1, 12))
        frefs = self.array(frefs, (N, 12))
        caps = self.array(caps, (N, 4))
        T = self.array(T, (N, 12, 12))
        bias = self.array(bias, (N, 12))
        tau_min, tau_max = (
            self.array(a, (N, 12))
            for a in (tau_min, tau_max)
        )

        if np.any(caps < 0) or np.any(tau_min >= tau_max):
            raise ValueError("Invalid QP bounds")

        self.A.data[self.ad_slots] = -Ad.reshape(N, 144)
        self.A.data[self.bd_slots] = -Bd.reshape(N, 144)
        self.A.data[self.torque_slots] = T.reshape(N, 144)

        bounds = np.r_[x, cd.ravel()]
        self.lower[:self.offset] = bounds
        self.upper[:self.offset] = bounds

        self.upper[self.fz_rows] = caps.ravel()
        self.lower[self.force_end:] = (
            tau_min - bias
        ).ravel()
        self.upper[self.force_end:] = (
            tau_max - bias
        ).ravel()

        q = np.r_[
            (-2.0 * self.weights * refs).ravel(),
            (-2.0 * self.R * frefs).ravel(),
        ]

        self.solver.update(
            Ax=self.A.data,
            q=q,
            l=self.lower,
            u=self.upper,
        )
        result = self.solver.solve(raise_error=False)

        self.last_info = result.info
        self.wall_ms = 1000.0 * (
            time.perf_counter() - started
        )

        if (
            result.info.status != "solved"
            or result.x is None
            or not np.isfinite(result.x).all()
        ):
            raise RuntimeError(
                f"QP failed: {result.info.status}"
            )

        az = self.A @ result.x
        self.last_violation = max(
            0.0,
            float(np.max(self.lower - az)),
            float(np.max(az - self.upper)),
        )

        if self.last_violation > 2e-4:
            raise RuntimeError(
                f"QP constraint violation: "
                f"{self.last_violation:.3g}"
            )

        self.predicted_states = result.x[
            :self.offset
        ].reshape(N + 1, 12).copy()

        self.predicted_forces = result.x[
            self.offset:
        ].reshape(N, 4, 3).copy()

        return self.predicted_forces[0].copy()


class MPCControl:
    def __init__(self, sim, trajectory):
        self.sim, self.trajectory = sim, trajectory
        self.qp = PersistentMPC()
        self.dt = 0.02

        self.low = sim.model.actuator_ctrlrange[:, 0] + 2.0
        self.high = sim.model.actuator_ctrlrange[:, 1] - 2.0

        trajectory.weight = (
            -sim.model.body_subtreemass[sim.trunk_id]
            * sim.model.opt.gravity[2]
        )

    def torque_map(self, q, dq):
        sim = self.sim
        model, data = sim.model, sim.data
        T = np.zeros((12, 12))

        for foot, geom in enumerate(sim.foot_geom_ids):
            point = data.geom_xpos[geom].copy()
            point[2] -= model.geom_size[geom, 0]

            mujoco.mj_jac(
                model,
                data,
                sim.jacp,
                sim.jacr,
                point,
                int(model.geom_bodyid[geom]),
            )

            T[:, 3 * foot:3 * foot + 3] = (
                -sim.jacp[:, sim.v_indices].T
            )

        bias = (
            data.qfrc_bias[sim.v_indices]
            + 100.0 * (
                q - data.qpos[sim.q_indices]
            )
            + 3.0 * (
                dq - data.qvel[sim.v_indices]
            )
        )

        return T, bias

    def solve(self, t, gate=np.inf, frozen=False):
        sim = self.sim
        model, data = sim.model, sim.data

        x = sim.read_state()
        q, dq = self.trajectory.posture(t, frozen)
        T, bias = self.torque_map(q, dq)

        com = data.subtree_com[sim.trunk_id].copy()
        contacts = data.geom_xpos[
            sim.foot_geom_ids
        ].copy()
        contacts[:, 2] -= model.geom_size[
            sim.foot_geom_ids, 0
        ]

        inertia = robot_inertia(
            model, data, sim.trunk_id, com
        )

        Ad, Bd, cd = build_mpc_model(
            model.body_subtreemass[sim.trunk_id],
            inertia,
            contacts,
            com,
            model.opt.gravity,
            dt=self.dt,
        )

        refs, frefs, caps = self.trajectory.preview(
            t, self.qp.N, self.dt, gate, frozen
        )

        return self.qp.solve(
            x, Ad, Bd, cd, refs, frefs,
            caps, T, bias, self.low, self.high,
        )

class WalkSimulation(StandingSimulation):
    def __init__(
        self, model_path,
        vx=0.001, vy=0.0, cycles=2,
    ):
        super().__init__(model_path)

        self.plan = GaitPlan(
            self.data.geom_xpos[self.foot_geom_ids],
            self.x_ref[3:6],
            vx,
            vy,
            cycles,
        )
        self.trajectory = Trajectory(self, self.plan)
        self.control = MPCControl(self, self.trajectory)

        self.t, self.gate_index = 0.0, 0
        self.held, self.waited = False, 0.0
        self.dirty = True
        self.missing_contact_time = 0.0
        self.solve_times = []
        self.status = "not solved"

    def gate_ready(self, gate):
        _, kind, step = gate

        loads = foot_ground_loads(self)
        feet = self.data.geom_xpos[self.foot_geom_ids]
        support = [
            i for i in range(4) if i != step.leg
        ]

        margin = triangle_margin(
            self.data.subtree_com[self.trunk_id, :2],
            feet[support, :2],
        )

        if np.min(loads[support]) < 5.0 or margin < 0.025:
            return False

        geom = self.foot_geom_ids[step.leg]
        clearance = (
            feet[step.leg, 2]
            - self.model.geom_size[geom, 0]
        )

        if kind == "unload":
            return (
                loads[step.leg] < 10.0
                and np.max(
                    np.abs(self.forces[step.leg])
                ) < 1e-3
                and np.sum(loads[support])
                > 0.85 * self.trajectory.weight
                and abs(self.read_state()[11]) < 0.03
            )

        if kind == "clear":
            return (
                clearance > 0.025
                and abs(loads[step.leg]) < 0.1
            )

        floor = self.model.geom("floor").id
        body = self.model.geom_bodyid[geom]

        contact = any(
            (
                c.geom1 == floor
                and self.model.geom_bodyid[c.geom2] == body
            )
            or (
                c.geom2 == floor
                and self.model.geom_bodyid[c.geom1] == body
            )
            for c in self.data.contact[:self.data.ncon]
        )

        return (
            contact
            and loads[step.leg] > 0.5
            and clearance < 0.003
        )

    def advance_clock(self):
        dt = self.model.opt.timestep

        if self.gate_index < len(self.plan.gates):
            gate = self.plan.gates[self.gate_index]

            if (
                self.held
                or self.t + dt >= gate[0] - 1e-10
            ):
                self.t = gate[0]
                arriving = not self.held
                self.held = not self.gate_ready(gate)
                self.dirty = (
                    self.dirty
                    or arriving
                    or not self.held
                )

                if self.held:
                    self.waited += dt

                    if self.waited > 2.0:
                        raise RuntimeError(
                            f"Contact gate timeout: "
                            f"{gate[1]} "
                            f"{LEG_NAMES[gate[2].leg]}"
                        )
                else:
                    self.gate_index += 1
                    self.waited = 0.0

                return

        self.t = min(
            self.t + dt, self.plan.duration
        )

    def step(self):
        model, data = self.model, self.data
        mujoco.mj_forward(model, data)

        self.applied_t = self.t

        gate = (
            self.plan.gates[self.gate_index][0]
            if self.gate_index < len(self.plan.gates)
            else np.inf
        )

        if (
            self.step_count % self.steps_per_mpc == 0
            or self.dirty
        ):
            started = time.perf_counter()

            try:
                self.forces = self.control.solve(
                    self.t, gate, self.held
                )
            except RuntimeError as error:
                raise RuntimeError(
                    f"STOP at {self.t:.3f}s: {error}"
                ) from error

            self.solve_times.append(
                1000.0 * (
                    time.perf_counter() - started
                )
            )

            self.status = "solved"
            self.mpc_updates += 1
            self.dirty = False

        q, dq = self.trajectory.posture(
            self.t, self.held
        )
        T, bias = self.control.torque_map(q, dq)
        torque = T @ self.forces.ravel() + bias

        limits = model.actuator_ctrlrange

        if (
            not np.isfinite(torque).all()
            or np.any(torque < limits[:, 0] - 1e-6)
            or np.any(torque > limits[:, 1] + 1e-6)
        ):
            raise RuntimeError(
                "STOP: applied torque would exceed actuator limits"
            )

        self.unclipped_torque = torque.copy()
        data.ctrl[:] = np.clip(
            torque, limits[:, 0], limits[:, 1]
        )

        mujoco.mj_step(model, data)
        self.step_count += 1
        mujoco.mj_forward(model, data)

        state = self.read_state()

        if (
            not np.isfinite(data.qpos).all()
            or not np.isfinite(state).all()
            or data.qpos[2] < 0.15
            or np.max(np.abs(state[:2]))
            > np.radians(10.0)
        ):
            raise RuntimeError(
                "STOP: unstable or non-finite robot state"
            )

        if self.t > 3.0:
            _, _, _, active, _ = self.plan.sample(self.t)
            required = np.ones(4, dtype=bool)

            if (
                active is not None
                and active.unload_start
                <= self.t < active.reload_end
            ):
                required[active.leg] = False

            missing = (
                np.min(
                    foot_ground_loads(self)[required]
                ) < 0.5
            )

            self.missing_contact_time = (
                self.missing_contact_time
                + model.opt.timestep
                if missing
                else 0.0
            )

            if self.missing_contact_time > 0.1:
                raise RuntimeError(
                    "STOP: lost a required stance contact"
                )

        self.advance_clock()

def run_case(
    vx=0.001,
    vy=0.0,
    cycles=2,
    headless=True,
    push=0.0,
    verbose=True,
):
    path = (
        Path(__file__).parent
        / "models/unitree_a1/scene_mpc.xml"
    )

    sim = WalkSimulation(path, vx, vy, cycles)
    baseline = sim.x_ref[3:6].copy()

    tilt = torque = drift = inactive = 0.0
    minimum_clearance = np.inf
    minimum_margin = np.inf
    maximum_swing_load = 0.0
    checked_steps = set()

    context = (
        nullcontext(None)
        if headless
        else mujoco.viewer.launch_passive(
            sim.model, sim.data
        )
    )

    if verbose:
        print(
            f"Planning complete: "
            f"{sim.plan.duration:.1f}s, "
            f"IK={sim.trajectory.ik_error:.2e}m"
        )

    with context as viewer:
        while sim.t < sim.plan.duration - 1e-9:
            if (
                viewer is not None
                and not viewer.is_running()
            ):
                raise RuntimeError(
                    "Simulation closed before completion"
                )

            started = time.perf_counter()
            sim.data.xfrc_applied[sim.trunk_id] = 0.0

            if sim.plan.steps:
                push_start = (
                    sim.plan.steps[0].lift_end + 0.3
                )

                if push_start <= sim.t < push_start + 0.2:
                    sim.data.xfrc_applied[
                        sim.trunk_id, 0
                    ] = push

            sim.step()
            state = sim.read_state()

            torque = max(
                torque,
                float(np.max(
                    np.abs(sim.unclipped_torque)
                )),
            )
            tilt = max(
                tilt,
                float(np.max(np.abs(state[:2]))),
            )

            if sim.t < 3.0:
                baseline = state[3:6].copy()

            feet_goal, _, _, active, phase = (
                sim.plan.sample(sim.applied_t)
            )
            feet = sim.data.geom_xpos[
                sim.foot_geom_ids
            ]
            stance = np.ones(4, dtype=bool)

            if (
                active is not None
                and active.lift_start
                <= sim.applied_t
                < active.reload_start
            ):
                stance[active.leg] = False

                inactive = max(
                    inactive,
                    float(np.max(np.abs(
                        sim.forces[active.leg]
                    ))),
                )

                if (
                    active.lift_end + 0.15
                    <= sim.applied_t
                    < active.advance_end
                ):
                    loads = foot_ground_loads(sim)
                    support = [
                        i for i in range(4)
                        if i != active.leg
                    ]

                    clearance = (
                        feet[active.leg, 2]
                        - sim.model.geom_size[
                            sim.foot_geom_ids[active.leg],
                            0,
                        ]
                    )
                    margin = triangle_margin(
                        state[3:5],
                        feet[support, :2],
                    )

                    minimum_clearance = min(
                        minimum_clearance, clearance
                    )
                    minimum_margin = min(
                        minimum_margin, margin
                    )
                    maximum_swing_load = max(
                        maximum_swing_load,
                        abs(float(loads[active.leg])),
                    )
                    checked_steps.add(active.start)

            drift = max(
                drift,
                float(np.max(np.linalg.norm(
                    feet[stance, :2]
                    - feet_goal[stance, :2],
                    axis=1,
                ))),
            )

            if viewer is not None:
                viewer.sync()

                remaining = (
                    sim.model.opt.timestep
                    - (time.perf_counter() - started)
                )

                if remaining > 0.0:
                    time.sleep(remaining)

    state = sim.read_state()
    loads = foot_ground_loads(sim)
    feet = sim.data.geom_xpos[sim.foot_geom_ids]

    com_error = np.linalg.norm(
        state[3:5] - sim.plan.final_com[:2]
    )
    foot_error = np.max(np.linalg.norm(
        feet[:, :2] - sim.plan.final_feet[:, :2],
        axis=1,
    ))

    expected = (
        sim.plan.final_com[:2] - sim.plan.com0[:2]
    )
    displacement_error = np.linalg.norm(
        state[3:5] - baseline[:2] - expected
    )

    passed = (
        len(checked_steps) == len(sim.plan.steps)
        and com_error < 0.012
        and displacement_error < 0.015
        and foot_error < 0.012
        and drift < 0.015
        and inactive < 1e-3
        and np.degrees(tilt) < 3.0
        and minimum_clearance > 0.025
        and minimum_margin > 0.025
        and maximum_swing_load < 0.1
        and np.min(loads) > 5.0
        and np.linalg.norm(state[9:11]) < 0.01
    )

    result = dict(
        passed=bool(passed),
        physical_s=round(sim.data.time, 3),
        steps=len(checked_steps),
        com_error_mm=float(com_error * 1000),
        foot_error_mm=float(foot_error * 1000),
        stance_error_mm=drift * 1000,
        tilt_deg=float(np.degrees(tilt)),
        max_torque_nm=torque,
        inactive_force_n=inactive,
        qp_p99_ms=float(np.percentile(
            sim.solve_times, 99
        )),
        qp_max_ms=max(sim.solve_times),
        updates=sim.mpc_updates,
    )

    if verbose:
        print(result)
        print(
            "MPC BASELINE PASSED"
            if passed
            else "MPC BASELINE FAILED"
        )

    return result


def solver_checks():
    qp = PersistentMPC(horizon=3)

    x = np.zeros(12)
    x[5] = 0.25

    feet = np.array([
        [.18, -.13, 0],
        [.18, .13, 0],
        [-.18, -.13, 0],
        [-.18, .13, 0],
    ])

    Ad, Bd, cd = build_mpc_model(
        12.453,
        np.diag([.14, .37, .40]),
        feet,
        x[3:6],
        [0, 0, -9.81],
    )

    refs = np.tile(x, (4, 1))
    fref = np.zeros((3, 12))
    fref[:, 2::3] = 12.453 * 9.81 / 4

    caps = np.full((3, 4), 100.0)
    upper = np.full(12, 33.5)
    upper[2] = 25.0

    pattern = (
        qp.A.indptr.copy(),
        qp.A.indices.copy(),
    )

    qp.solve(
        x, Ad, Bd, cd, refs, fref, caps,
        np.eye(12), np.zeros(12),
        -33.5, upper,
    )

    forces = qp.predicted_forces
    states = qp.predicted_states

    assert np.max(forces[:, 0, 2]) <= 25.0 + 2e-4

    for k in range(3):
        residual = (
            states[k + 1]
            - Ad @ states[k]
            - Bd @ forces[k].ravel()
            - cd
        )
        assert np.max(np.abs(residual)) < 2e-4

    caps[:, 0] = 0.0

    qp.solve(
        x, Ad, Bd, cd, refs, fref, caps,
        np.zeros((12, 12)), np.zeros(12),
        -33.5, 33.5,
    )

    assert np.max(
        np.abs(qp.predicted_forces[:, 0])
    ) < 1e-3

    assert np.array_equal(
        qp.A.indptr, pattern[0]
    )
    assert np.array_equal(
        qp.A.indices, pattern[1]
    )

    try:
        qp.solve(
            x, Ad, Bd, cd, refs, fref, caps,
            np.zeros((12, 12)),
            np.full(12, 1000.0),
            -33.5, 33.5,
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError(
            "Infeasible torque constraints were accepted"
        )


def stop_checks():
    path = (
        Path(__file__).parent
        / "models/unitree_a1/scene_mpc.xml"
    )
    sim = WalkSimulation(path, cycles=1)

    sim.held, sim.waited = True, 2.0
    sim.gate_ready = lambda gate: False

    try:
        sim.advance_clock()
    except RuntimeError:
        pass
    else:
        raise AssertionError(
            "Contact timeout was ignored"
        )

    def failed_solve(*args):
        raise RuntimeError("injected solver failure")

    sim.control.solve = failed_solve
    before = sim.data.time

    try:
        sim.step()
    except RuntimeError:
        pass
    else:
        raise AssertionError(
            "Solver failure was ignored"
        )

    assert sim.data.time == before


def verify():
    solver_checks()
    stop_checks()
    print("Solver and stop checks PASSED")

    cases = [
        ("stand", 0.0, 0.0, 1, 0.0),
        ("forward", 0.001, 0.0, 2, 0.0),
        ("backward", -0.001, 0.0, 1, 0.0),
        ("left", 0.0, 0.001, 1, 0.0),
        ("right", 0.0, -0.001, 1, 0.0),
        ("push", 0.001, 0.0, 1, 20.0),
    ]

    passed = True

    for name, vx, vy, cycles, push in cases:
        try:
            result = run_case(
                vx, vy, cycles, push=push
            )
            passed &= result["passed"]

            print(
                name,
                "PASSED" if result["passed"] else "FAILED",
            )
        except (RuntimeError, ValueError) as error:
            passed = False
            print(name, "FAILED:", error)

    print(
        "MPC VERIFICATION PASSED"
        if passed
        else "MPC VERIFICATION FAILED"
    )

    return passed


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--vx", type=float, default=0.001
    )
    parser.add_argument(
        "--vy", type=float, default=0.0
    )
    parser.add_argument(
        "--cycles", type=int, default=2
    )
    parser.add_argument(
        "--headless", action="store_true"
    )
    parser.add_argument(
        "--verify", action="store_true"
    )

    args = parser.parse_args()

    try:
        ok = (
            verify()
            if args.verify
            else run_case(
                args.vx,
                args.vy,
                args.cycles,
                args.headless,
            )["passed"]
        )
    except (RuntimeError, ValueError) as error:
        print("MPC BASELINE FAILED:", error)
        ok = False

    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
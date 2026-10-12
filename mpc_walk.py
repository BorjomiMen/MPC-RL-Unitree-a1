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
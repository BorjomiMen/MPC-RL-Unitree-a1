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
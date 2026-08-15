#!/usr/bin/env python3
"""Validate the converted M20 + Piper USD in a minimal physics scene."""

from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--steps", type=int, default=120, help="Number of physics steps to run.")
parser.add_argument(
    "--hard-exit",
    action="store_true",
    help="Exit directly after validation (workaround for Isaac Sim shutdown hangs in headless mode).",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
from pxr import Usd, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import build_simulation_context

from custom_envs.assets.m20_piper_single import (
    ARM_JOINT_NAMES,
    GRIPPER_JOINT_NAMES,
    M20_PIPER_SINGLE_USD,
    DEEPROBOTICS_M20_PIPER_SINGLE_CFG,
)


M20_JOINT_NAMES = [
    "fl_hipx_joint", "fl_hipy_joint", "fl_knee_joint", "fl_wheel_joint",
    "fr_hipx_joint", "fr_hipy_joint", "fr_knee_joint", "fr_wheel_joint",
    "hl_hipx_joint", "hl_hipy_joint", "hl_knee_joint", "hl_wheel_joint",
    "hr_hipx_joint", "hr_hipy_joint", "hr_knee_joint", "hr_wheel_joint",
]
EXPECTED_JOINTS = set(M20_JOINT_NAMES + ARM_JOINT_NAMES + GRIPPER_JOINT_NAMES)


def validate_usd() -> None:
    """Check composition before loading the asset through Isaac Lab."""
    stage = Usd.Stage.Open(M20_PIPER_SINGLE_USD)
    if stage is None:
        raise RuntimeError(f"Unable to open USD: {M20_PIPER_SINGLE_USD}")
    roots = [str(prim.GetPath()) for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.ArticulationRootAPI)]
    if len(roots) != 1:
        raise AssertionError(f"Expected exactly one ArticulationRootAPI, found {len(roots)}: {roots}")
    print(f"USD articulation root: {roots[0]}", flush=True)


def validate_physics() -> None:
    """Load one articulation, command joint1, and run a short finite-value test."""
    print(f"Creating minimal simulation context on {args.device}", flush=True)
    with build_simulation_context(device=args.device, auto_add_lighting=False, add_ground_plane=False) as sim:
        print("Simulation context created", flush=True)
        sim_utils.create_prim("/World/Env_0", "Xform")
        print("Spawning articulation", flush=True)
        robot = Articulation(DEEPROBOTICS_M20_PIPER_SINGLE_CFG.replace(prim_path="/World/Env_.*/Robot"))
        print("Articulation spawned; resetting simulation", flush=True)
        sim.reset()
        print("Simulation reset completed", flush=True)

        if not robot.is_initialized:
            raise AssertionError("Articulation failed to initialize")
        if robot.is_fixed_base:
            raise AssertionError("Expected a floating-base articulation")
        if robot.num_joints != 24:
            raise AssertionError(f"Expected 24 DOF, got {robot.num_joints}")
        if set(robot.joint_names) != EXPECTED_JOINTS:
            missing = EXPECTED_JOINTS - set(robot.joint_names)
            extra = set(robot.joint_names) - EXPECTED_JOINTS
            raise AssertionError(f"Joint-name mismatch: missing={missing}, extra={extra}")

        joint1_index = robot.joint_names.index("joint1")
        initial_joint1 = float(robot.data.joint_pos[0, joint1_index])
        targets = robot.data.default_joint_pos.clone()
        targets[:, joint1_index] = min(initial_joint1 + 0.15, 2.0)

        for _ in range(args.steps):
            robot.set_joint_position_target(targets)
            robot.write_data_to_sim()
            sim.step()
            robot.update(sim.cfg.dt)

        if not torch.isfinite(robot.data.root_state_w).all():
            raise AssertionError("Non-finite root state after simulation")
        if not torch.isfinite(robot.data.joint_pos).all():
            raise AssertionError("Non-finite joint state after simulation")
        final_joint1 = float(robot.data.joint_pos[0, joint1_index])
        if abs(final_joint1 - initial_joint1) < 1.0e-3:
            raise AssertionError(
                f"joint1 did not respond: initial={initial_joint1:.6f}, final={final_joint1:.6f}"
            )

        print(f"DOF count: {robot.num_joints}", flush=True)
        print(f"Body count: {robot.num_bodies}", flush=True)
        print(f"Joint names: {robot.joint_names}", flush=True)
        print(f"Body names: {robot.body_names}", flush=True)
        print(f"joint1 response: {initial_joint1:.6f} -> {final_joint1:.6f}", flush=True)
        print("M20 + Piper single-articulation validation PASSED", flush=True)
        if args.hard_exit:
            # Exit before SimulationContext.__exit__ runs: Isaac Sim 5.1 can
            # block there indefinitely in headless mode on this workstation.
            os._exit(0)


try:
    validate_usd()
    validate_physics()
finally:
    simulation_app.close()
"""
Robot arm controller — uses Jacobian-transpose IK for positioning
and direct ctrl commands to velocity/position actuators.

Motion is paced against the wall clock and eased in/out so the cycle reads
like a real machine: approach, slow descent, gradual grip, slow lift,
transit, slow release.
"""

import logging
import os
import time

import numpy as np
import mujoco

from simulation.mujoco_env import MuJoCoEnv
from config import MuJoCoConfig

logger = logging.getLogger(__name__)

# Wall-clock pacing. 1.0 = real time, 2.0 = half speed, 0 = as fast as possible
# (used by the headless self-check). Tune here, not in config.py.
REALTIME = float(os.environ.get("ROBOT_REALTIME", "1.0"))

# Joint-velocity caps (rad/s) per phase — the arm is slow where it matters.
TRANSIT_SPEED = 1.40    # free air, nothing near
APPROACH_SPEED = 0.40   # descending onto a product / into a bin
LIFT_SPEED = 0.70       # carrying a product

# Gripper ctrl: inner finger faces sit at 0.038 + q, products are 40mm wide.
GRIP_OPEN = 0.0
GRIP_CLOSED = -0.024   # 5mm past contact so the servo keeps a real squeeze on


class RobotArm:
    """
    High-level pick-and-place using Jacobian-transpose IK.
    Works with the sorting_scene.xml arm definition.
    """

    BIN_SITES = {
        "A": "bin_a",
        "B": "bin_b",
        "C": "bin_c",
        "D": "bin_d",
    }

    def __init__(self, env: MuJoCoEnv, config: MuJoCoConfig):
        self.env = env
        self.config = config
        self.model = env.model
        self.data = env.data
        self._wall = time.perf_counter()

        # Cache joint and actuator IDs
        self.joint_ids = []
        self.joint_dof_ids = []
        for name in ["joint1", "joint2", "joint3", "joint4"]:
            jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self.joint_ids.append(jnt_id)
            self.joint_dof_ids.append(self.model.joint(jnt_id).dofadr[0])

        self.act_ids = []
        for name in ["joint1", "joint2", "joint3", "joint4"]:
            self.act_ids.append(
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            )

        self.grip_l_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "grip_l"
        )
        self.grip_r_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "grip_r"
        )

        self.ee_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "ee"
        )

        # Cache bin positions
        self.bin_positions = {}
        for bin_id, site_name in self.BIN_SITES.items():
            self.bin_positions[bin_id] = self.env.get_site_pos(site_name)

        logger.info("Robot arm initialized — %d bins cached", len(self.bin_positions))

    def pick_and_place(self, item_body_name: str, bin_id: str):
        """Full pick-and-place cycle, paced for a watching human."""
        item_pos = self.env.get_body_pos(item_body_name)
        bin_pos = self.bin_positions[bin_id]

        logger.info(f"Pick-and-place: {item_body_name} → Bin {bin_id}")

        # 1. Open the gripper before going anywhere near the product
        self._set_grip(open=True)

        # 2. Stand off above the product
        self._move_to(item_pos + [0, 0, 0.16], TRANSIT_SPEED)

        # 3. Two-stage descent — the second stage is a crawl
        self._move_to(item_pos + [0, 0, 0.055], APPROACH_SPEED)
        self._move_to(item_pos, APPROACH_SPEED * 0.6, tol=0.006)
        self._dwell(0.3)

        # 4. Squeeze closed, then let the grip load up
        self._set_grip(open=False)
        self._dwell(0.4)

        # 5. Lift straight up, slowly, so the product does not swing out
        self._move_to(item_pos + [0, 0, 0.07], LIFT_SPEED * 0.5)
        self._move_to(item_pos + [0, 0, 0.18], LIFT_SPEED)

        # 6. Carry across to the bin at height
        self._move_to(bin_pos + [0, 0, 0.22], LIFT_SPEED)

        # 7. Lower into the bin mouth
        self._move_to(bin_pos + [0, 0, 0.11], APPROACH_SPEED)
        self._dwell(0.3)

        # 8. Release and let it drop the last centimetre
        self._set_grip(open=True)
        self._dwell(0.5)

        # 9. Clear the bin, then back to a neutral pose
        self._move_to(bin_pos + [0, 0, 0.22], APPROACH_SPEED)
        self._move_to(np.array([0.0, 0.25, 0.35]), TRANSIT_SPEED, tol=0.02)

        logger.info(f"Placed {item_body_name} in Bin {bin_id}")

    def _move_to(self, target: np.ndarray, speed: float = None,
                 max_steps: int = 1200, tol: float = 0.004):
        """
        Move the grasp centre to target with Jacobian-transpose IK.
        Velocity eases in at the start and decays with distance at the end,
        so there is no teleport-and-stop.
        """
        target = np.asarray(target, dtype=np.float64)
        cap = self.config.move_speed if speed is None else speed
        dq_prev = np.zeros(len(self.act_ids))

        for step in range(max_steps):
            error = target - self.data.site_xpos[self.ee_site_id]
            dist = float(np.linalg.norm(error))
            if dist < tol:
                break

            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_site_id)
            J = jacp[:, self.joint_dof_ids]

            dq = J.T @ error
            norm = np.linalg.norm(dq)
            if norm < 1e-9:
                break

            # decelerate into the target, ease out of the start
            v = min(cap, 4.0 * dist + 0.06) * min(1.0, (step + 1) / 20.0)
            dq = dq / norm * v

            # slew limit — keeps the first frames from snapping
            dq = dq_prev + np.clip(dq - dq_prev, -0.05, 0.05)
            dq_prev = dq

            for i, act_id in enumerate(self.act_ids):
                self.data.ctrl[act_id] = dq[i]

            self._tick(4)

        # coast to a stop instead of cutting the ctrl dead
        for scale in (0.5, 0.2, 0.0):
            for i, act_id in enumerate(self.act_ids):
                self.data.ctrl[act_id] = dq_prev[i] * scale
            self._tick(6)

    def _set_grip(self, open: bool, seconds: float = 0.6):
        """Drive the fingers to their target over `seconds` — no snapping shut."""
        target = GRIP_OPEN if open else GRIP_CLOSED
        start = self.data.ctrl[self.grip_l_id]
        n = max(1, int(seconds / (4 * self.model.opt.timestep)))

        for k in range(1, n + 1):
            # smoothstep so the fingers start and finish gently
            t = k / n
            q = start + (target - start) * (t * t * (3 - 2 * t))
            self.data.ctrl[self.grip_l_id] = q
            self.data.ctrl[self.grip_r_id] = q
            self._tick(4)

    def _dwell(self, seconds: float):
        """Hold still — the beat that makes the motion read as deliberate."""
        for act_id in self.act_ids:
            self.data.ctrl[act_id] = 0.0
        for _ in range(max(1, int(seconds / (4 * self.model.opt.timestep)))):
            self._tick(4)

    def _tick(self, n: int = 4):
        """Step the sim, refresh the viewer, and hold the wall clock to REALTIME."""
        self.env.step(n_steps=n)
        self.env.sync_viewer()
        if REALTIME <= 0:
            return
        self._wall += n * self.model.opt.timestep * REALTIME
        slack = self._wall - time.perf_counter()
        if slack > 0:
            time.sleep(slack)
        elif slack < -0.2:
            self._wall = time.perf_counter()   # fell behind, resync

    def go_home(self):
        self._move_to(np.array([0.0, 0.25, 0.35]), TRANSIT_SPEED, tol=0.02)
        self._set_grip(open=True)


if __name__ == "__main__":
    # Self-check: python -m simulation.robot_arm
    # Runs the whole cycle headless and fails if any product misses its bin.
    import config as _cfg

    REALTIME = 0.0                      # no wall-clock pacing for the check
    _env = MuJoCoEnv(_cfg.MuJoCoConfig())
    _arm = RobotArm(_env, _env.config)
    _env.step(500)

    for _body, _bin in [("item_0", "A"), ("item_1", "B"), ("item_2", "C"),
                        ("item_3", "D"), ("item_4", "B")]:
        _t0 = _env.data.time
        _arm.pick_and_place(_body, _bin)
        _env.step(300)
        _pos, _want = _env.get_body_pos(_body), _arm.bin_positions[_bin]
        print(f"{_body} -> bin {_bin}  {np.round(_pos, 3)}  cycle={_env.data.time - _t0:.1f}s sim")
        assert abs(_pos[0] - _want[0]) < 0.08 and abs(_pos[1] - _want[1]) < 0.08             and _pos[2] < 0.09, f"{_body} missed bin {_bin}: {_pos}"

    print("all products landed in their bins")

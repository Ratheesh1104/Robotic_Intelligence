import logging
from typing import Optional

import numpy as np
import mujoco
import mujoco.viewer

from config import MuJoCoConfig

logger = logging.getLogger(__name__)

_LABEL_DOT = np.array([0.006, 0.0, 0.0])
_IDENTITY = np.eye(3).flatten()
_DOT_RGBA = np.array([1.0, 1.0, 1.0, 0.85], dtype=np.float32)


class MuJoCoEnv:
    def __init__(self, config: MuJoCoConfig):
        self.config = config
        self.model = mujoco.MjModel.from_xml_path(config.scene_xml)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = config.time_step
        self._viewer: Optional[mujoco.viewer.Handle] = None
        self._labels = []

        # Reset to home keyframe if it exists
        self._reset_to_home()

        logger.info(f"MuJoCo env loaded from {config.scene_xml}")

    def _reset_to_home(self):
        """Try to reset to the 'home' keyframe so arm starts bent."""
        try:
            home_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_KEY, "home"
            )
            if home_id >= 0:
                mujoco.mj_resetDataKeyframe(self.model, self.data, home_id)
                mujoco.mj_forward(self.model, self.data)
                logger.info("Reset to 'home' keyframe — arm starts in bent pose")
        except Exception:
            logger.warning("No 'home' keyframe found, using default pose")

    @property
    def viewer_running(self) -> bool:
        return self._viewer is not None and self._viewer.is_running()

    def launch_viewer(self):
        self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self._labels = self._collect_labels()
        self._draw_labels()
        logger.info("MuJoCo viewer launched — %d labels", len(self._labels))

    def step(self, n_steps: int = 1):
        for _ in range(n_steps):
            mujoco.mj_step(self.model, self.data)

    def sync_viewer(self):
        if self._viewer is not None:
            self._draw_labels()
            self._viewer.sync()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    def get_body_pos(self, name: str):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise KeyError(f"no body named {name!r} in {self.config.scene_xml}")
        return self.data.xpos[body_id].copy()

    def get_site_pos(self, name: str):
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        if site_id < 0:
            raise KeyError(f"no site named {name!r} in {self.config.scene_xml}")
        return self.data.site_xpos[site_id].copy()

    # ─── on-screen labels ────────────────────────────────────────────
    # Text comes from <custom><text name="label_<thing>"> in the scene XML, so
    # the scene owns the wording. Bin labels are fixed in place; product labels
    # ride along with the body so you can see what the gripper is carrying.

    def _custom_text(self, name: str) -> Optional[str]:
        tid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_TEXT, name)
        if tid < 0:
            return None
        adr = self.model.text_adr[tid]
        raw = bytes(self.model.text_data[adr:adr + self.model.text_size[tid]])
        return raw.split(b"\x00")[0].decode("utf-8", "replace")

    def _collect_labels(self):
        """[(text, body_id_or_None, fixed_pos_or_None, z_offset)]"""
        labels = []

        for site_name in ("bin_a", "bin_b", "bin_c", "bin_d"):
            text = self._custom_text(f"label_{site_name}")
            if text:
                labels.append((text, None, self.get_site_pos(site_name), 0.14))

        for body_id in range(self.model.nbody):
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if not body_name or not body_name.startswith("item_"):
                continue
            text = self._custom_text(f"label_{body_name}")
            if text:
                labels.append((text, body_id, None, 0.06))

        return labels

    def _draw_labels(self):
        scn = getattr(self._viewer, "user_scn", None)
        if scn is None or not self._labels:
            return

        scn.ngeom = 0
        for text, body_id, fixed_pos, dz in self._labels:
            if scn.ngeom >= scn.maxgeom:
                break
            pos = (fixed_pos if body_id is None else self.data.xpos[body_id]).copy()
            pos[2] += dz
            geom = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(
                geom, mujoco.mjtGeom.mjGEOM_SPHERE,
                _LABEL_DOT, pos, _IDENTITY, _DOT_RGBA,
            )
            geom.label = text
            scn.ngeom += 1

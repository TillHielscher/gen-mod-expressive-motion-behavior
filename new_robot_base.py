"""
Abstract Robot Base

Defines the interface every robot module must implement.

Key design decisions
--------------------
* **Virtual session** is robot-agnostic and lives in ``new_core.py``.  The robot
  only supplies the URDF path, initial joint angles, and a method to convert
  internal DMP state → virtual-session joint array.
* **Real-robot session** is entirely owned by the robot subclass.  The core
  never touches hardware directly — it calls ``create_real_session()`` once
  and ``execute_on_real_robot()`` each control tick.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


class RobotBase(ABC):
    """Interface that every robot implementation must satisfy."""

    def __init__(self, robot_dir: str) -> None:
        self.robot_dir = Path(robot_dir)
        self.config = self._load_config()

    # ── config loading ───────────────────────────────────────────────────────

    def _load_config(self) -> dict:
        for candidate in (
            self.robot_dir / "robot_data.yaml",
            self.robot_dir / f"{self.robot_dir.name}.yaml",
        ):
            if candidate.exists():
                with open(candidate, "r") as f:
                    return yaml.safe_load(f)
        raise FileNotFoundError(
            f"No config found in {self.robot_dir}  "
            f"(tried robot_data.yaml / {self.robot_dir.name}.yaml)"
        )

    # ── metadata (concrete, config-driven) ───────────────────────────────────

    def get_capabilities(self) -> str:
        return self.config.get("capabilities", "")

    def get_primitive_lib(self) -> dict:
        return self.config.get("primitive_lib", {})

    def get_parameter_ranges(self) -> dict:
        return self.config.get("parameter_ranges", {})

    def get_primitive_path(self) -> str:
        if "primitives_path" in self.config:
            return self.config["primitives_path"]
        inside = self.robot_dir / f"{self.robot_dir.name}_primitives"
        if inside.exists():
            return str(inside)
        return str(inside)  # fallback — will error later if missing

    def get_robot_description_path(self) -> str:
        return str(self.robot_dir / f"{self.robot_dir.name}.yaml")

    # ── abstract: identity ───────────────────────────────────────────────────

    @abstractmethod
    def get_joint_names(self) -> List[str]:
        """Ordered list of joint names in the internal DMP representation."""
        ...

    @abstractmethod
    def get_joint_index(self, joint_name: str) -> int:
        """Index of *joint_name* in the internal DMP representation."""
        ...

    @abstractmethod
    def get_urdf_path(self) -> str:
        """Absolute path to the URDF used for virtual visualisation."""
        ...

    def get_initial_joint_angles(self) -> Dict[str, float]:
        """Return {joint_name: radians} for the virtual session's starting pose.

        Override in the robot subclass to supply a meaningful default pose.
        Returning an empty dict means all joints start at 0.
        """
        return {}

    # ── abstract: virtual execution ──────────────────────────────────────────

    @abstractmethod
    def execute_state_on_virtual_robot(self, virtual_session, state) -> None:
        """Push a DMP state array to the virtual session.

        The robot subclass is responsible for any index reordering or
        coordinate conversion needed between the internal DMP representation
        and the URDF joint order.
        """
        ...

    # ── abstract: real-robot session (owned by robot module) ─────────────────

    def create_real_session(self) -> None:
        """Create & store the hardware connection.

        Called once by ``Core.run()`` when ``use_real_robot=True``.
        The default implementation raises — override in the robot subclass.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement create_real_session()"
        )

    def execute_on_real_robot(self, state) -> None:
        """Send a DMP state array to the real hardware.

        Called every control tick by the core when ``use_real_robot=True``.
        The default implementation raises — override in the robot subclass.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement execute_on_real_robot()"
        )

    # ── optional: floating-base robots ───────────────────────────────────────

    def compute_base_transform(self, joint_array):
        """For floating-base robots (quadrupeds), return (position, quaternion).

        Fixed-base robots should leave this as-is (returns ``None``).
        """
        return None

    # ── trajectory translation (optional) ────────────────────────────────────

    def translate_trajectory_to_internal(self, external_traj):
        """Convert robot-native trajectory → internal DMP format."""
        raise NotImplementedError

    def translate_trajectory_to_external(self, internal_traj):
        """Convert internal DMP format → robot-native trajectory."""
        raise NotImplementedError

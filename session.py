"""
Session – robot base interface and Viser-based virtual session.

``RobotBase``
    Abstract interface that every robot module must implement.  Includes
    automatic kinematic analysis to derive ``Follow_Through_Data`` from
    the URDF when it is not already present in the robot YAML.

``ViserView``
    Unified Viser server providing the 3D URDF viewer **and** all GUI panels
    (context inputs, planner status, sequence visualization).
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from collections import defaultdict
from math import cos, sin
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import numpy as np
import viser
import yaml
from viser.extras import ViserUrdf
from yourdfpy import URDF

if TYPE_CHECKING:
    from core import Core

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Kinematic relation analysis (auto-compute Follow_Through_Data)
# ---------------------------------------------------------------------------

def _rpy_to_matrix(r: float, p: float, y: float) -> np.ndarray:
    """Convert roll-pitch-yaw angles to a 3×3 rotation matrix."""
    Rz = np.array([[cos(y), -sin(y), 0],
                   [sin(y),  cos(y), 0],
                   [0,       0,      1]])
    Ry = np.array([[cos(p), 0, sin(p)],
                   [0,       1, 0],
                   [-sin(p), 0, cos(p)]])
    Rx = np.array([[1, 0, 0],
                   [0, cos(r), -sin(r)],
                   [0, sin(r),  cos(r)]])
    return Rz @ Ry @ Rx


def _axis_angle_matrix(axis: np.ndarray, theta: float) -> np.ndarray:
    """Rodrigues formula for rotation about an arbitrary axis."""
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s, C = cos(theta), sin(theta), 1 - cos(theta)
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


class _URDFKinematicAnalyzer:
    """Parse a URDF and discover axis-alignment relations between revolute joints.

    Used internally by :class:`RobotBase` to auto-generate ``Follow_Through_Data``
    when it is not already present in the robot YAML.
    """

    def __init__(self, urdf_path: str, active_joint_names: Optional[List[str]] = None) -> None:
        self.joints: dict = {}
        self.child_map: dict[str, list[str]] = defaultdict(list)
        self.parent_map: dict[str, str] = {}
        self._active_joint_names = set(active_joint_names) if active_joint_names else None
        self._parse_urdf(urdf_path)

    # -- URDF parsing ---------------------------------------------------------

    def _parse_urdf(self, urdf_path: str) -> None:
        tree = ET.parse(urdf_path)
        root = tree.getroot()
        for joint in root.findall("joint"):
            jname = joint.attrib["name"]
            jtype = joint.attrib["type"]
            if jtype != "revolute":
                continue

            parent = joint.find("parent").attrib["link"]
            child = joint.find("child").attrib["link"]

            origin_tag = joint.find("origin")
            rpy = ([float(x) for x in origin_tag.attrib.get("rpy", "0 0 0").split()]
                   if origin_tag is not None else [0.0, 0.0, 0.0])

            axis_tag = joint.find("axis")
            axis = np.array([float(x) for x in axis_tag.attrib.get("xyz", "0 0 1").split()])
            axis = axis / np.linalg.norm(axis)

            limit_tag = joint.find("limit")
            lower = float(limit_tag.attrib.get("lower", -np.pi)) if limit_tag is not None else -np.pi
            upper = float(limit_tag.attrib.get("upper", np.pi)) if limit_tag is not None else np.pi

            self.joints[jname] = {
                "type": jtype,
                "parent": parent,
                "child": child,
                "axis_local": axis,
                "rpy": rpy,
                "lower": lower,
                "upper": upper,
            }
            self.child_map[parent].append(jname)
            self.parent_map[child] = jname

    # -- Forward kinematics (axis only) ----------------------------------------

    def _forward_axis(self, joint_name: str, joint_values: dict) -> np.ndarray:
        """Compute the world-frame axis of *joint_name* given *joint_values*."""
        chain: list[str] = []
        cur = joint_name
        while cur in self.joints:
            chain.append(cur)
            parent_link = self.joints[cur]["parent"]
            if parent_link not in self.parent_map:
                break
            cur = self.parent_map[parent_link]
        chain.reverse()

        T = np.eye(3)
        for j in chain:
            info = self.joints[j]
            T = T @ _rpy_to_matrix(*info["rpy"])
            if info["type"] == "revolute":
                T = T @ _axis_angle_matrix(info["axis_local"], joint_values.get(j, 0.0))
        axis_global = T @ self.joints[joint_name]["axis_local"]
        return axis_global / np.linalg.norm(axis_global)

    # -- Relation discovery ----------------------------------------------------

    def _is_active(self, joint_name: str) -> bool:
        """Return True if *joint_name* belongs to the robot's active joints."""
        if self._active_joint_names is None:
            return True
        return joint_name in self._active_joint_names

    def find_relations(self, samples: int = 200, tol: float = 0.95) -> list[dict]:
        """Discover axis-aligned joint relations.

        Returns a list of dicts compatible with the ``Follow_Through_Data``
        YAML format (keys: target, source, inverse, condition, lower_limit,
        upper_limit).
        """
        relations: list[dict] = []

        # 1. Conditional chains: source → condition → target
        for cond_name, cond in self.joints.items():
            if cond["type"] != "revolute":
                continue
            children = self.child_map.get(cond["child"], [])
            if not children:
                continue
            parent_joint = self.parent_map.get(cond["parent"])
            if parent_joint is None:
                continue

            for target in children:
                source = parent_joint
                if not (self._is_active(source) and self._is_active(target) and self._is_active(cond_name)):
                    continue

                cond_vals = np.linspace(cond["lower"], cond["upper"], samples)
                current_interval = None

                for val in cond_vals:
                    joint_values = {cond_name: val}
                    src_axis = self._forward_axis(source, joint_values)
                    tgt_axis = self._forward_axis(target, joint_values)
                    dot = float(np.dot(src_axis, tgt_axis))

                    if abs(dot) >= tol:
                        if current_interval is None:
                            current_interval = [val, val]
                        else:
                            current_interval[1] = val
                    else:
                        if current_interval is not None:
                            relations.append({
                                "target": target,
                                "source": source,
                                "inverse": False,
                                "condition": cond_name,
                                "lower_limit": float(current_interval[0]),
                                "upper_limit": float(current_interval[1]),
                            })
                            current_interval = None

                if current_interval is not None:
                    relations.append({
                        "target": target,
                        "source": source,
                        "inverse": False,
                        "condition": cond_name,
                        "lower_limit": float(current_interval[0]),
                        "upper_limit": float(current_interval[1]),
                    })

        # 2. Direct parent→child pairs (no condition)
        for joint_name, joint in self.joints.items():
            parent_joint = self.parent_map.get(joint["parent"])
            if parent_joint is None:
                continue
            source, target = parent_joint, joint_name
            if not (self._is_active(source) and self._is_active(target)):
                continue

            src_axis = self._forward_axis(source, {})
            tgt_axis = self._forward_axis(target, {})
            dot = float(np.dot(src_axis, tgt_axis))
            if abs(dot) >= tol:
                relations.append({
                    "target": target,
                    "source": source,
                    "inverse": bool(dot < 0),
                    "condition": None,
                    "lower_limit": None,
                    "upper_limit": None,
                })

        return relations


# ---------------------------------------------------------------------------
# RobotBase
# ---------------------------------------------------------------------------

class RobotBase(ABC):
    """Interface that every robot implementation must satisfy.

    * **Virtual session** is robot-agnostic and lives in ``ViserView``.
      The robot only supplies the URDF path, initial joint angles, and a
      method to convert internal DMP state → virtual-session joint array.
    * **Real-robot session** is entirely owned by the robot subclass.
    * **Follow_Through_Data** is automatically computed from the URDF
      when not present in the robot YAML.  Subclasses should call
      ``self._ensure_follow_through_data()`` at the end of their
      ``__init__`` (after ``self.urdf_path`` is set).  The computed
      relations are written back to the YAML so the analysis only
      runs once.
    """

    def __init__(self, robot_dir: str) -> None:
        self.robot_dir = Path(robot_dir)
        self.config = self._load_config()

    # -- Config loading --------------------------------------------------------

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

    # -- Metadata (concrete, config-driven) ------------------------------------

    def get_capabilities(self) -> str:
        return self.config.get("capabilities", "")

    def get_primitive_lib(self) -> dict:
        return self.config.get("primitive_lib", {})

    def get_parameter_ranges(self) -> dict:
        return self.config.get("parameter_ranges", {})

    def _ensure_follow_through_data(self) -> None:
        """Auto-compute ``Follow_Through_Data`` from the URDF if absent.

        Call this at the end of the subclass ``__init__`` (after
        ``self.urdf_path`` and joint definitions are ready).  If the
        robot YAML already contains ``Follow_Through_Data`` the method
        returns immediately.  Otherwise it analyses the URDF, computes
        the kinematic relations, stores them in ``self.config``, **and**
        writes them back to the YAML file so the computation only
        happens once.
        """
        ranges = self.config.setdefault("parameter_ranges", {})
        if "Follow_Through_Data" in ranges:
            logger.debug("Follow_Through_Data already present in config — skipping auto-computation.")
            return

        try:
            urdf_path = self.get_urdf_path()
            joint_names = self.get_joint_names()
        except Exception:
            logger.warning("Cannot auto-compute Follow_Through_Data: URDF path or joint names unavailable.")
            return

        if not Path(urdf_path).exists():
            logger.warning("Cannot auto-compute Follow_Through_Data: URDF not found at %s", urdf_path)
            return

        logger.info("Auto-computing Follow_Through_Data from URDF: %s", urdf_path)
        try:
            analyzer = _URDFKinematicAnalyzer(urdf_path, active_joint_names=joint_names)
            relations = analyzer.find_relations()
        except Exception:
            logger.exception("Failed to auto-compute Follow_Through_Data.")
            return

        if not relations:
            logger.info("No kinematic relations found for Follow_Through_Data.")
            return

        follow_data: dict = {}
        for idx, rel in enumerate(relations, 1):
            follow_data[f"relation{idx}"] = {
                "target": rel["target"],
                "source": rel["source"],
                "inverse": rel["inverse"],
                "condition": rel["condition"] if rel["condition"] is not None else "None",
                "lower_limit": rel["lower_limit"],
                "upper_limit": rel["upper_limit"],
            }
        ranges["Follow_Through_Data"] = follow_data
        logger.info("Auto-computed %d Follow_Through relation(s).", len(relations))

        # Persist to the YAML file so it is only computed once.
        self._write_follow_through_to_yaml(follow_data)

    def _write_follow_through_to_yaml(self, follow_data: dict) -> None:
        """Insert ``Follow_Through_Data`` into the robot YAML without reformatting.

        Locates the ``Follow_Through:`` block inside ``parameter_ranges``
        (the one with ``min``/``max`` sub-keys) and inserts the new block
        right after its ``max:`` line, preserving every other byte of the
        original file.
        """
        yaml_path = self._config_path()
        if yaml_path is None:
            logger.warning("Could not locate robot YAML — computed data kept in memory only.")
            return

        try:
            with open(yaml_path, "r") as f:
                lines = f.readlines()

            # --- find insertion point: after the "max:" line of Follow_Through ---
            import re
            insert_idx: Optional[int] = None
            in_follow_through = False
            for i, line in enumerate(lines):
                stripped = line.strip()
                # Match "Follow_Through:" as a top-level parameter_ranges key
                # (2-space indent), but NOT "Follow_Through_Data:"
                if re.match(r"^\s{2}Follow_Through:\s*$", line):
                    in_follow_through = True
                    continue
                if in_follow_through:
                    if stripped.startswith("max:"):
                        insert_idx = i + 1
                        break
                    # If we hit another key at the same indent before "max:",
                    # the structure is unexpected — bail out.
                    if stripped and not stripped.startswith("min:") and not stripped.startswith("#"):
                        break

            if insert_idx is None:
                logger.warning(
                    "Could not find Follow_Through max: line in %s — "
                    "computed data kept in memory only.", yaml_path,
                )
                return

            # --- format the block as YAML text (2-space base indent) ---
            block_lines = ["  Follow_Through_Data:\n"]
            for rel_key, rel in follow_data.items():
                block_lines.append(f"    {rel_key}:\n")
                for k, v in rel.items():
                    block_lines.append(f"      {k}: {v}\n")

            lines[insert_idx:insert_idx] = block_lines

            with open(yaml_path, "w") as f:
                f.writelines(lines)
            logger.info("Follow_Through_Data written to %s", yaml_path)
        except Exception:
            logger.exception("Failed to write Follow_Through_Data to %s", yaml_path)

    def _config_path(self) -> Optional[Path]:
        """Return the path to the robot YAML config file, or *None*."""
        for candidate in (
            self.robot_dir / "robot_data.yaml",
            self.robot_dir / f"{self.robot_dir.name}.yaml",
        ):
            if candidate.exists():
                return candidate
        return None

    def get_primitive_path(self) -> str:
        if "primitives_path" in self.config:
            return self.config["primitives_path"]
        inside = self.robot_dir / f"{self.robot_dir.name}_primitives"
        return str(inside)

    def get_robot_description_path(self) -> str:
        return str(self.robot_dir / f"{self.robot_dir.name}.yaml")

    # -- Abstract: identity ----------------------------------------------------

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

    # -- Abstract: virtual execution -------------------------------------------

    @abstractmethod
    def execute_state_on_virtual_robot(self, virtual_session, state, primitive_name: str = "") -> None:
        """Push a DMP state array to the virtual session.

        The robot subclass is responsible for any index reordering or
        coordinate conversion needed between the internal DMP representation
        and the URDF joint order.
        """
        ...

    # -- Optional: real-robot session (owned by robot module) ------------------

    def create_real_session(self) -> None:
        """Create & store the hardware connection.

        Called once by ``Core.run()`` when ``use_real_robot=True``.
        Override in the robot subclass to implement real-robot support.
        """
        pass

    def execute_state_on_real_robot(self, state) -> None:
        """Send a DMP state array to the real hardware.

        Called every control tick by the core when ``use_real_robot=True``.
        Override in the robot subclass to implement real-robot support.
        """
        pass

    # -- Optional: real-time data ----------------------------------------------

    def prepare_handle_rt(self) -> None:
        """Prepare state needed by :meth:`handle_rt`.

        Called once during ``__init__``.  Must set ``self.rt_goal_indices``
        to a list of joint indices whose DMP goals are managed in real
        time (these are preserved across block transitions).

        Default: no RT-managed joints.
        """
        self.rt_goal_indices: list[int] = []

    def handle_rt(self, block) -> None:
        """Handle real-time modulation of the current motion block.

        Called every tick (~20 Hz) by the core's RT loop.  Override in
        the robot subclass to implement robot-specific RT behaviour
        (e.g. human tracking, gaze following, obstacle avoidance).

        Args:
            block: The current ``TimelineBlock``.
        """
        pass

    # -- Optional: floating-base robots ----------------------------------------

    def compute_base_transform(self, joint_array):
        """For floating-base robots (quadrupeds), return (position, quaternion).

        Fixed-base robots should leave this as-is (returns ``None``).
        """
        return None


# ---------------------------------------------------------------------------
# ViserView
# ---------------------------------------------------------------------------

class ViserView:
    """Viser server: 3D URDF scene + GUI panels.

    Args:
        urdf_path: Path to the robot URDF.
        initial_joint_angles: ``{joint_name: radians}`` initial pose.
        principle_ranges: ``{"Anticipation": {"scale_range": [0, 10]}, …}``.
        port: Viser server port.
    """

    def __init__(
        self,
        urdf_path: Union[str, Path],
        initial_joint_angles: Optional[Dict[str, float]] = None,
        principle_ranges: Optional[dict] = None,
        port: int = 8088,
    ) -> None:
        # -- 3D scene ------------------------------------------------------
        self.urdf_path = Path(urdf_path)
        if not self.urdf_path.exists():
            raise FileNotFoundError(f"URDF not found: {self.urdf_path}")

        self.server = viser.ViserServer(port=port)
        yourdf = URDF.load(str(self.urdf_path))

        self.viser_urdf = ViserUrdf(
            self.server,
            urdf_or_path=yourdf,
            load_meshes=True,
        )

        self.initial_config = list(
            np.clip(
                [initial_joint_angles.get(name, 0.0)
                 for name in self.viser_urdf.get_actuated_joint_limits()],
                [lo if lo is not None else -np.pi
                 for lo, _ in self.viser_urdf.get_actuated_joint_limits().values()],
                [hi if hi is not None else np.pi
                 for _, hi in self.viser_urdf.get_actuated_joint_limits().values()],
            )
        )
        self.viser_urdf.update_cfg(np.array(self.initial_config))

        scene = self.viser_urdf._urdf.scene or self.viser_urdf._urdf.collision_scene
        self.server.scene.add_grid(
            "/grid", width=2000.0, height=2000.0,
            position=(0.0, 0.0, scene.bounds[0, 2] if scene is not None else 0.0),
            cell_thickness=1000.0, shadow_opacity=1.0,
        )
        self.server.scene.enable_default_lights()

        # -- GUI state -----------------------------------------------------
        self._principle_ranges = principle_ranges or {}
        self._core: Optional[Core] = None  # set via attach_core()

        # Sequence GUI handles
        self._now_playing_section = None
        self._plan_section = None
        self._now_playing_folder = None
        self._now_playing_label: Optional[str] = None
        self._plan_entries: list = []       # [(folder, sliders, name, plan_idx)]
        self._last_seen_block = None

        # Status display
        self._status_handle = None

    # -- Lifecycle ---------------------------------------------------------

    def attach_core(self, core: Core) -> None:
        """Wire back-references needed for context push and slider re-modulation."""
        self._core = core
        self._add_status_display()
        self._add_context_inputs()
        self._init_now_playing()

    # -- 3D scene helpers (called by robot modules) ------------------------

    def set_cfg_array(self, values) -> None:
        self.viser_urdf.update_cfg(np.asarray(values))

    def set_base_transform(self, position: np.ndarray, quaternion: np.ndarray) -> None:
        wxyz = np.array([quaternion[3], quaternion[0], quaternion[1], quaternion[2]])
        for attr in ("_visual_root_frame", "_collision_root_frame"):
            frame = getattr(self.viser_urdf, attr, None)
            if frame is not None:
                frame.position = tuple(position)
                frame.wxyz = tuple(wxyz)

    # -- Context inputs ----------------------------------------------------

    def _add_context_inputs(self) -> None:
        gui = self.server.gui

        with gui.add_folder("Context", expand_by_default=True):
            text_field = gui.add_text(label="Text", initial_value="")
            image_field = gui.add_text(label="Image path", initial_value="")
            audio_field = gui.add_text(label="Audio path", initial_value="")
            button = gui.add_button(label="Send Context")

        @button.on_click
        def _(_):
            ctx: dict = {}
            if text_field.value.strip():
                ctx["text"] = text_field.value.strip()
            if image_field.value.strip():
                ctx["image"] = image_field.value.strip()
            if audio_field.value.strip():
                ctx["audio"] = audio_field.value.strip()
            if ctx:
                self._core.context_store.push(ctx)
                text_field.value = ""
                image_field.value = ""
                audio_field.value = ""

    # -- Status display ----------------------------------------------------

    def _add_status_display(self) -> None:
        gui = self.server.gui
        self._status_handle = gui.add_markdown("⏳ **Awaiting context**")

    def set_status(self, text: str) -> None:
        """Update the planner status shown in the webview."""
        if self._status_handle is not None:
            self._status_handle.content = text

    # -- Now Playing (always visible, even for idle blocks) ----------------

    def _init_now_playing(self) -> None:
        """Create the persistent Now Playing section showing the current block."""
        gui = self.server.gui
        self._now_playing_section = gui.add_folder("Now Playing", expand_by_default=True)
        block = self._core.timeline.get_current_block()
        name = block.name_identifier if block else "idle"
        self._show_now_playing(name)
        self._last_seen_block = block

    def _show_now_playing(self, label: str, slider_values: Optional[dict] = None) -> None:
        """Replace the inner Now Playing folder with a new one."""
        gui = self.server.gui
        if self._now_playing_folder is not None:
            self._now_playing_folder.remove()
            self._now_playing_folder = None

        with self._now_playing_section:
            folder = gui.add_folder(label, expand_by_default=True)
            if slider_values:
                with folder:
                    for principle, info in self._principle_ranges.items():
                        lo, hi = info["scale_range"]
                        initial = slider_values.get(principle, 0)
                        gui.add_slider(
                            label=principle,
                            min=lo, max=hi, step=1,
                            initial_value=initial,
                            disabled=True,
                        )
        self._now_playing_folder = folder
        self._now_playing_label = label

    # -- Sequence visualisation --------------------------------------------

    def build_sequence(self, unmapped_seq: list[dict]) -> None:
        """Create the 'Plan' section with block folders."""
        self._clear_plan()
        gui = self.server.gui

        self._plan_section = gui.add_folder("Plan", expand_by_default=True)

        with self._plan_section:
            for idx, unmapped in enumerate(unmapped_seq):
                entry = self._create_block_folder(gui, idx, unmapped)
                self._plan_entries.append(entry)

        self._last_seen_block = self._core.timeline.get_current_block()

    def _create_block_folder(self, gui, idx: int, unmapped: dict):
        """Create a single block folder with principle sliders."""
        name = unmapped.get("motion_primitive", "?")
        folder = gui.add_folder(f"#{idx + 1}  {name}", expand_by_default=True)
        sliders: dict = {}

        with folder:
            for principle, info in self._principle_ranges.items():
                lo, hi = info["scale_range"]
                initial = unmapped.get(principle, 0)
                # clip initial to valid range
                initial = max(min(initial, hi), lo)
                slider = gui.add_slider(
                    label=principle,
                    min=lo, max=hi, step=1,
                    initial_value=initial,
                )
                sliders[principle] = slider

                def _on_change(_, _idx=idx, _sliders=sliders):
                    self._on_slider_update(_idx, _sliders)
                slider.on_update(_on_change)

        return (folder, sliders, name, idx)

    def advance_sequence(self) -> None:
        """Move the next plan entry into 'Now Playing', removing the old one."""
        if not self._plan_entries:
            # Plan exhausted — show the current block name (likely idle)
            block = self._core.timeline.get_current_block()
            name = block.name_identifier if block else "idle"
            self._show_now_playing(name)
            return

        old_folder, sliders, prim_name, idx = self._plan_entries.pop(0)
        current_values = {name: s.value for name, s in sliders.items()}
        old_folder.remove()

        self._show_now_playing(f"#{idx + 1}  {prim_name}", slider_values=current_values)

    def _clear_plan(self) -> None:
        """Remove the Plan section (Now Playing persists)."""
        for folder, _, _, _ in self._plan_entries:
            folder.remove()
        self._plan_entries.clear()
        if self._plan_section is not None:
            self._plan_section.remove()
            self._plan_section = None

    def _on_slider_update(self, plan_idx: int, sliders: dict) -> None:
        """Re-map and re-apply modulation when a slider value changes."""
        unmapped = {name: int(s.value) for name, s in sliders.items()}
        remapped = self._core.planner._map_principles_to_parameters(unmapped)

        plan = self._core.timeline.plan
        if plan_idx < len(plan):
            self._core._apply_modulation(plan[plan_idx], remapped)
            logger.info("[view] Slider update → re-modulated block #%d", plan_idx + 1)

    # -- Tick (called from Core's planner-trigger loop) --------------------

    def tick(self) -> None:
        """Detect timeline block changes and update Now Playing."""
        block = self._core.timeline.get_current_block()
        if block is not self._last_seen_block:
            self._last_seen_block = block
            self.advance_sequence()

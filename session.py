"""
Session – robot base interface and Viser-based virtual session.

``RobotBase``
    Abstract interface that every robot module must implement.

``ViserView``
    Unified Viser server providing the 3D URDF viewer **and** all GUI panels
    (context inputs, planner status, sequence visualization).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
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
# RobotBase
# ---------------------------------------------------------------------------

class RobotBase(ABC):
    """Interface that every robot implementation must satisfy.

    * **Virtual session** is robot-agnostic and lives in ``ViserView``.
      The robot only supplies the URDF path, initial joint angles, and a
      method to convert internal DMP state → virtual-session joint array.
    * **Real-robot session** is entirely owned by the robot subclass.
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

    def handle_rt(self, block) -> None:
        """Handle real-time modulation of the current motion block.

        Called every tick (~20 Hz) by the core's RT loop.  Override in
        the robot subclass to implement robot-specific RT behaviour
        (e.g. human tracking, gaze following, obstacle avoidance).

        Args:
            block: The current ``TimelineBlock``.
        """
        pass  # no RT handling by default

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

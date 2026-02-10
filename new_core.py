"""
Core – orchestrates timeline, planner, context, and visualisation.

The **virtual session** (Viser-based URDF viewer) is always part of the core
because it is robot-agnostic: it only needs a URDF path and optional initial
joint angles, both of which come from ``RobotBase``.

The **real-robot session** is *not* managed here.  Each robot module creates
and owns its own hardware session via ``RobotBase.create_real_session()`` /
``RobotBase.execute_on_real_robot()``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import viser
from viser.extras import ViserUrdf
from yourdfpy import URDF

from new_block import TimelineBlock
from new_context import ContextStore
from new_planner import Planner
from new_timeline import Timeline

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Virtual Session (always linked to core)
# ═══════════════════════════════════════════════════════════════════════════════

class VirtualSession:
    """Robot-agnostic URDF viewer built on Viser.

    All robot-specific configuration (initial joint angles, etc.) is passed in
    from the outside — nothing is hardcoded here.
    """

    def __init__(
        self,
        urdf_path: Union[str, Path],
        initial_joint_angles: Optional[Dict[str, float]] = None,
        port: int = 8088,
    ) -> None:
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
                [initial_joint_angles.get(name, 0.0) for name in self.viser_urdf.get_actuated_joint_limits()],
                [lo if lo is not None else -np.pi for lo, _ in self.viser_urdf.get_actuated_joint_limits().values()],
                [hi if hi is not None else np.pi for _, hi in self.viser_urdf.get_actuated_joint_limits().values()],
            )
        )
        self.viser_urdf.update_cfg(np.array(self.initial_config))

        # Grid & lights
        scene = self.viser_urdf._urdf.scene or self.viser_urdf._urdf.collision_scene
        self.server.scene.add_grid(
            "/grid", width=2000.0, height=2000.0,
            position=(0.0, 0.0, scene.bounds[0, 2] if scene is not None else 0.0),
            cell_thickness=1000.0, shadow_opacity=1.0,
        )
        self.server.scene.enable_default_lights()

    # ── joint access ──────────────────────────────────────────────────────────

    def get_joint_names(self) -> List[str]:
        return [s.label for s in self.slider_handles]

    def set_cfg_array(self, values) -> None:
        self.viser_urdf.update_cfg(np.asarray(values))

    def set_base_transform(self, position: np.ndarray, quaternion: np.ndarray) -> None:
        wxyz = np.array([quaternion[3], quaternion[0], quaternion[1], quaternion[2]])
        for attr in ("_visual_root_frame", "_collision_root_frame"):
            frame = getattr(self.viser_urdf, attr, None)
            if frame is not None:
                frame.position = tuple(position)
                frame.wxyz = tuple(wxyz)


# ═══════════════════════════════════════════════════════════════════════════════
# Core
# ═══════════════════════════════════════════════════════════════════════════════

class Core:
    """Main orchestrator: controller loop · planner trigger · context listener."""

    def __init__(
        self,
        robot,
        *,
        use_real_robot: bool = False,
        use_virtual_robot: bool = True,
        short_pipeline: bool = True,
        modulate: bool = True,
        prompt_data_path: str = "prompts_v4.yaml",
        debug: bool = False,
    ) -> None:
        self.robot = robot
        self.use_real_robot = use_real_robot
        self.use_virtual_robot = use_virtual_robot
        self.short_pipeline = short_pipeline
        self.modulate = modulate

        # Logging
        level = logging.DEBUG if debug else logging.INFO
        logging.basicConfig(level=level)

        # Sub-systems
        self.timeline = Timeline(
            robot.get_primitive_path(),
            robot.get_robot_description_path(),
        )
        self.planner = Planner(robot, prompt_data_path)
        self.context_store = ContextStore()
        self._executor = ThreadPoolExecutor(max_workers=1)

        # Populated during run()
        self.virtual_session: Optional[VirtualSession] = None
        self._markdown: Optional[object] = None

    # ── async run ─────────────────────────────────────────────────────────────

    async def run(self) -> None:
        # Real-robot session — owned by the robot module
        if self.use_real_robot:
            self.robot.create_real_session()

        # Virtual session — always owned by core
        if self.use_virtual_robot:
            initial_angles = self.robot.get_initial_joint_angles()
            self.virtual_session = VirtualSession(
                urdf_path=self.robot.get_urdf_path(),
                initial_joint_angles=initial_angles,
            )
            self._markdown = self.virtual_session.server.gui.add_markdown(
                content="Timeline Plan: (empty)"
            )
            self._add_context_gui()
            await self._wait_for_client()

        await asyncio.gather(
            self._safe(self._controller_loop(), "controller_loop"),
            self._safe(self._context_listener(), "context_listener"),
            self._safe(self._planner_trigger_loop(), "planner_trigger_loop"),
            self._safe(self._rt_data_loop(), "rt_data_loop"),
        )

    # ── real-time data loop ──────────────────────────────────────────────────

    async def _rt_data_loop(self, hz: int = 20) -> None:
        """Core-owned async loop that calls the robot's handle_rt()."""
        interval = 1.0 / hz
        while True:
            block = self.timeline.get_current_block()
            if block is not None:
                self.robot.handle_rt(block)
            await asyncio.sleep(interval)

    # ── controller loop ──────────────────────────────────────────────────────

    async def _controller_loop(self, hz: int = 60) -> None:
        interval = 1.0 / hz
        cycle_even = True

        while True:
            t0 = time.time()
            block = self.timeline.get_current_block()
            if block:
                block.step()
                state_y = block.dmp.get_state()["y"]

                if cycle_even:
                    if self.use_real_robot:
                        self.robot.execute_on_real_robot(state_y)
                    if self.use_virtual_robot:
                        self.robot.execute_state_on_virtual_robot(self.virtual_session, state_y)

                if block.is_complete():
                    self.timeline.advance_block()

            cycle_even = not cycle_even
            await asyncio.sleep(max(0.0, interval - (time.time() - t0)))

    # ── context listener (stdin) ─────────────────────────────────────────────

    async def _context_listener(self) -> None:
        loop = asyncio.get_event_loop()
        while True:
            raw = await loop.run_in_executor(None, input, "Enter context (ENTER to confirm):\n")
            text = raw.strip()
            if text:
                self.context_store.push({"text": text})
            await asyncio.sleep(0.5)

    # ── planner trigger ──────────────────────────────────────────────────────

    async def _planner_trigger_loop(self) -> None:
        last_version = -1

        while True:
            self._update_status_display()

            ctx = self.context_store.get()
            version = ctx.get("_version", -1)

            if version != last_version and ctx:
                last_version = version
                logger.info("[core] Planner triggered  context_v=%d", version)
                self._set_status("Planning …")

                loop = asyncio.get_event_loop()
                pipeline = self.planner.short_pipeline if self.short_pipeline else self.planner.long_pipeline
                mapped_seq, unmapped_seq = await loop.run_in_executor(self._executor, pipeline, ctx)

                logger.warning("[core] Unmapped sequence:\n%s", json.dumps(unmapped_seq, indent=2))

                self.timeline.clear_plan()
                for anim in mapped_seq:
                    block = TimelineBlock(
                        anim["motion_primitive"],
                        primitive_path=self.robot.get_primitive_path(),
                        idle_data_yaml_path=self.robot.get_robot_description_path(),
                    )
                    if self.modulate:
                        self._apply_modulation(block, anim)
                    self.timeline.append_block(block)

                logger.info("[core] New plan added  len=%d", len(self.timeline.plan))

            await asyncio.sleep(0.1)

    # ── modulation ───────────────────────────────────────────────────────────

    def _apply_modulation(self, block: TimelineBlock, anim: dict) -> None:
        follow_data_list = []
        for entry in anim.get("Follow_Through_Data", {}).values():
            row = [
                self.robot.get_joint_index(entry["target"]),
                self.robot.get_joint_index(entry["source"]),
                entry["inverse"],
            ]
            if "none" not in entry["condition"].lower():
                row += [
                    self.robot.get_joint_index(entry["condition"]),
                    entry["lower_limit"],
                    entry["upper_limit"],
                ]
            follow_data_list.append(np.array(row))

        block.dmp.set_principle_parameters(
            p_ant=anim["Anticipation"],
            p_follow=anim["Follow_Through"],
            p_arc=anim["Arcs"],
            p_slow=anim["Slow_In_Slow_Out"],
            p_progression=["fast", "fast", "fast"],
            p_time=anim["Timing"],
            p_exa=anim["Exaggeration"],
            p_follow_data=follow_data_list,
        )

    # ── GUI helpers ──────────────────────────────────────────────────────────

    def _add_context_gui(self) -> None:
        gui = self.virtual_session.server.gui

        with gui.add_folder("Context", expand_by_default=True):
            text_field = gui.add_text(
                label="Text", initial_value=""
            )
            image_field = gui.add_text(
                label="Image path", initial_value=""
            )
            audio_field = gui.add_text(
                label="Audio path", initial_value=""
            )
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
                self.context_store.push(ctx)
                text_field.value = ""
                image_field.value = ""
                audio_field.value = ""

    def _update_status_display(self) -> None:
        if self._markdown is None:
            return
        block = self.timeline.get_current_block()
        name = block.name_identifier if block else "—"
        plan_names = self.timeline.get_plan_names()
        self._markdown.content = (
            f'Executing **"{name}"**\n\n'
            f"Remaining plan: {plan_names if plan_names else '(empty)'}"
        )

    def _set_status(self, text: str) -> None:
        if self._markdown is not None:
            self._markdown.content = text

    async def _wait_for_client(self) -> None:
        while True:
            if self.virtual_session.server.get_clients():
                logger.info("Viser client connected.")
                return
            logger.info("Waiting for Viser client …")
            await asyncio.sleep(0.5)

    @staticmethod
    async def _safe(coro, name: str) -> None:
        try:
            await coro
        except Exception:
            logger.exception("Exception in %s", name)
            raise

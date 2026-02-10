"""
Core – orchestrates timeline, planner, and context.

The **ViserView** (Viser-based URDF viewer + GUI) is created here but lives
in ``session.py``.  It is robot-agnostic: it only needs a URDF path and
optional initial joint angles, both of which come from ``RobotBase``.

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
from typing import Optional

import numpy as np

from context import ContextStore
from planner import Planner
from session import ViserView
from timeline import Timeline, TimelineBlock

logger = logging.getLogger(__name__)


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

        # Principle scale ranges from prompt data (for GUI sliders)
        import yaml as _yaml
        with open(prompt_data_path) as _f:
            _pdata = _yaml.safe_load(_f)
        self._principle_ranges: dict = _pdata.get("principles", {})

        # Populated during run()
        self.view: Optional[ViserView] = None

    # ── async run ─────────────────────────────────────────────────────────────

    async def run(self) -> None:
        # Real-robot session — owned by the robot module
        if self.use_real_robot:
            self.robot.create_real_session()

        # Viser view — 3D scene + GUI panels
        if self.use_virtual_robot:
            initial_angles = self.robot.get_initial_joint_angles()
            self.view = ViserView(
                urdf_path=self.robot.get_urdf_path(),
                initial_joint_angles=initial_angles,
                principle_ranges=self._principle_ranges,
            )
            self.view.attach_core(self)
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
                        self.robot.execute_state_on_virtual_robot(
                            self.view, state_y, block.name_identifier
                        )

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
            if self.view:
                self.view.tick()

            ctx = self.context_store.get()
            version = ctx.get("_version", -1)

            if version != last_version and ctx:
                last_version = version
                logger.info("[core] Planner triggered  context_v=%d", version)

                if self.view:
                    self.view.set_status("⚙️ **Planning …**")

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

                # Build GUI blocks for the new sequence
                if self.view:
                    self.view.build_sequence(unmapped_seq)
                    self.view.set_status("⏳ **Awaiting context**")

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

    # ── utilities ──────────────────────────────────────────────────────────────

    async def _wait_for_client(self) -> None:
        while True:
            if self.view.server.get_clients():
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

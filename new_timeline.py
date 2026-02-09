"""
Motion Timeline

Maintains an ordered queue of ``TimelineBlock`` instances.
The controller pops the current block, steps it, and advances when complete.
"""

from __future__ import annotations

import logging
from copy import deepcopy

import numpy as np

from new_block import TimelineBlock

logger = logging.getLogger(__name__)


class Timeline:
    """FIFO queue of motion blocks with seamless joining."""

    def __init__(self, primitive_path: str, robot_description_path: str) -> None:
        self.primitive_path = primitive_path
        self.robot_description_path = robot_description_path

        self.plan: list[TimelineBlock] = []
        self.current_block: TimelineBlock = TimelineBlock(
            "idle",
            primitive_path=primitive_path,
            idle_data_yaml_path=robot_description_path,
        )

        # Real-time tracking offsets consumed by block.update_goal()
        self.rt_data: dict = {"x": 0.0, "y": 0.0}

    # -- queue operations ------------------------------------------------------

    def append_block(self, block: TimelineBlock) -> None:
        self.plan.append(block)
        if self.current_block is None:
            self.current_block = self.plan.pop(0)

    def clear_plan(self) -> None:
        self.plan.clear()

    # -- accessors -------------------------------------------------------------

    def get_current_block(self) -> TimelineBlock | None:
        return self.current_block

    def get_plan_names(self) -> list[str]:
        return [b.name_identifier for b in self.plan]

    # -- advance ---------------------------------------------------------------

    def advance_block(self) -> None:
        """Move to the next block, seamlessly joining DMP states."""
        last_state = None
        last_goal = None
        if self.current_block:
            last_state = deepcopy(self.current_block.dmp.get_state())
            last_goal = self.current_block.dmp.goal

        if self.plan:
            self.current_block = self.plan.pop(0)
            logger.info(
                "[timeline] Next block: %s  remaining: %d",
                self.current_block.name_identifier,
                len(self.plan),
            )
        else:
            self.current_block = TimelineBlock(
                "idle",
                primitive_path=self.primitive_path,
                idle_data_yaml_path=self.robot_description_path,
            )
            logger.info("[timeline] Plan empty → idle (%s)", self.current_block.name_identifier)

            # Add subtle randomisation to zero-motion idle primitives
            if self.current_block.name_identifier.lower().startswith("zero"):
                self.current_block.dmp.set_principle_parameters(p_rand=400)
                self.current_block.dmp.forcing_term.w_original = self.current_block.dmp.forcing_term.w
                self.current_block.dmp.set_principle_parameters(p_arc=40)
                n = self.current_block.dmp.n_dim
                exa = np.ones(n)
                exa[:min(5, n)] = [0.3, 0.3, 0.4, 0.2, 0.2][:min(5, n)]
                self.current_block.dmp.set_principle_parameters(p_exa=exa)

        # Seamless state joining
        if last_state is not None:
            self.current_block.dmp.y = last_state["y"]
            self.current_block.dmp.yd = last_state["yd"]
            self.current_block.dmp.ydd = last_state["ydd"]
            # Preserve tracked goal channels (head yaw / pitch when present)
            if last_goal is not None and len(last_goal) > 4:
                self.current_block.dmp.goal[3] = last_goal[3]
                self.current_block.dmp.goal[4] = last_goal[4]

        self.current_block.update_goal(self.rt_data)

"""
Timeline – motion block and FIFO queue.

``TimelineBlock``
    Wraps a single DMP motion primitive loaded from disk.

``Timeline``
    Maintains an ordered queue of blocks.  The controller pops the current
    block, steps it, and advances when complete.
"""

from __future__ import annotations

import logging
import os
import random
from copy import deepcopy
from typing import Optional

import numpy as np
import yaml

import animation_dmp

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# TimelineBlock
# ---------------------------------------------------------------------------

class TimelineBlock:
    """One motion primitive on the timeline, backed by an ``animation_dmp.DMP``."""

    def __init__(
        self,
        name_identifier: str,
        primitive_path: str,
        idle_data_yaml_path: str,
    ) -> None:
        self.name_identifier = name_identifier
        self.dmp = self._load_dmp(primitive_path, name_identifier, idle_data_yaml_path)
        self.dmp.init_state()

    # -- Runtime interface -----------------------------------------------------

    def step(self) -> None:
        self.dmp.step()

    def is_complete(self) -> bool:
        state = self.dmp.get_state()
        return state["t"] >= state["tau"]

    # -- Loading ---------------------------------------------------------------

    def _load_dmp(
        self,
        folder_path: str,
        name_identifier: str,
        idle_yaml_path: str,
    ) -> animation_dmp.DMP:
        """Load a DMP from *folder_path* whose base-name best matches *name_identifier*."""
        if not os.path.isdir(folder_path):
            raise ValueError(f"Primitive folder does not exist: {folder_path}")

        base_names = [f[:-5] for f in os.listdir(folder_path) if f.endswith(".json")]
        if not base_names:
            raise ValueError(f"No DMP files (.json) found in {folder_path}")

        chosen: Optional[str] = None

        # ---- special identifiers ----
        if name_identifier.lower() == "random":
            chosen = random.choice(base_names)

        elif name_identifier.lower() == "idle":
            with open(idle_yaml_path, "r") as f:
                idle_names = yaml.safe_load(f).get("idle_lib", [])
            if not idle_names:
                raise ValueError("No 'idle_lib' list in YAML.")
            name_identifier = random.choice(idle_names)
            self.name_identifier = name_identifier

        # ---- direct or fuzzy match ----
        if chosen is None:
            chosen = self._match_name(name_identifier, base_names)

        path = os.path.join(folder_path, chosen)
        return animation_dmp.DMP.load(path)

    @staticmethod
    def _match_name(identifier: str, base_names: list[str]) -> str:
        """Return the best-matching base name for *identifier*."""
        id_lower = identifier.lower()

        # exact match
        for bn in base_names:
            if bn.lower() == id_lower:
                return bn

        # fuzzy substring scoring
        scores: dict[str, float] = {}
        for bn in base_names:
            bn_lower = bn.lower()
            score = 0.0
            for i in range(len(id_lower)):
                for j in range(i + 1, len(id_lower) + 1):
                    if id_lower[i:j] in bn_lower:
                        score += len(id_lower[i:j]) ** 2
            score += len(set(id_lower) & set(bn_lower))
            score -= len(bn_lower) * 0.1
            scores[bn] = score

        return max(scores, key=scores.get)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Timeline
# ---------------------------------------------------------------------------

class Timeline:
    """FIFO queue of motion blocks with seamless joining."""

    def __init__(self, robot) -> None: 

        self.robot = robot       
            
        self.primitive_path = self.robot.get_primitive_path()
        self.robot_description_path = self.robot.get_robot_description_path()

        self.plan: list[TimelineBlock] = []
        self.current_block: TimelineBlock = TimelineBlock(
            "idle",
            primitive_path=self.primitive_path,
            idle_data_yaml_path=self.robot_description_path,
        )

    # -- Queue operations ------------------------------------------------------

    def append_block(self, block: TimelineBlock) -> None:
        self.plan.append(block)
        if self.current_block is None:
            self.current_block = self.plan.pop(0)

    def clear_plan(self) -> None:
        self.plan.clear()

    # -- Accessors -------------------------------------------------------------

    def get_current_block(self) -> TimelineBlock | None:
        return self.current_block

    def get_plan_names(self) -> list[str]:
        return [b.name_identifier for b in self.plan]

    # -- Advance ---------------------------------------------------------------

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

        # Seamless state joining
        if last_state is not None:
            self.current_block.dmp.y = last_state["y"]
            self.current_block.dmp.yd = last_state["yd"]
            self.current_block.dmp.ydd = last_state["ydd"]

            # Preserve RT-managed goal channels across block transitions
            if last_goal is not None:
                for idx in self.robot.rt_goal_indices:
                    if idx < len(last_goal):
                        self.current_block.dmp.goal[idx] = last_goal[idx]

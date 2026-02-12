"""
Kinova Gen3 Lite Robot Class

This module provides a robot-specific implementation for the Kinova Gen3 Lite robot arm.
It encapsulates all Gen3 Lite-specific functionality including trajectory translation,
joint mapping, and robot configuration.
"""

import sys
import numpy as np
from pathlib import Path

# Add parent directory to path to import robot_base
sys.path.insert(0, str(Path(__file__).parent.parent))
from session import RobotBase


class Gen3LiteRobot(RobotBase):
    """Kinova Gen3 Lite robot arm implementation.

    Handles Gen3 Lite-specific trajectory translation, joint mapping,
    and configuration management.  This is a 7-DOF robot arm.
    """

    # -- Definitions -----------------------------------------------------------
    
    # Gen3 Lite's default joint angles (in radians)
    DEFAULT_JOINT_ANGLES = np.array([0.0, -0.6, 1.9, -1.6, 2.0, 0.0, 0.5])
    
    # Joint names for the 7-DOF arm
    JOINT_NAMES = [
        "joint_1",  # Base rotation
        "joint_2",  # Shoulder
        "joint_3",  # Arm
        "joint_4",  # Forearm 1
        "joint_5",  # Forearm 2
        "joint_6",  # Wrist 1
        "joint_7",  # Wrist 2
    ]
    
    # Mapping from joint names to indices
    JOINT_NAME_TO_IDX = {name: idx for idx, name in enumerate(JOINT_NAMES)}
    
    def __init__(self, robot_dir="robot_gen3_lite"):
        """Initialize Gen3 Lite robot.

        Args:
            robot_dir: Path to the Gen3 Lite robot configuration directory.
        """
        super().__init__(robot_dir)
        # URDF path can be overridden, but defaults to description folder inside robot directory
        self.urdf_path = self.config.get(
            'urdf_path',
            str(self.robot_dir / 'robot_gen3_lite_description' / 'robot_gen3_lite.urdf')
        )

        self._ensure_follow_through_data()

    # -- Robot description related ---------------------------------------------
    
    def get_joint_names(self):
        """Return ordered list of Gen3 Lite joint names."""
        return self.JOINT_NAMES.copy()
    
    def get_joint_index(self, joint_name):
        """Return the index for the given joint name.

        Args:
            joint_name: Name of the joint (e.g. ``'joint_1'``).

        Returns:
            Index of the joint.

        Raises:
            ValueError: If *joint_name* is unknown.
        """
        if joint_name not in self.JOINT_NAME_TO_IDX:
            raise ValueError(f"Unknown joint name: {joint_name}")
        return self.JOINT_NAME_TO_IDX[joint_name]
    
    def get_initial_joint_angles(self):
        """Return default joint angles in radians for the virtual session."""
        return {name: float(self.DEFAULT_JOINT_ANGLES[i]) for i, name in enumerate(self.JOINT_NAMES)}
    
    def get_urdf_path(self):
        """Get path to URDF file."""
        return self.urdf_path
    
    # -- Real robot ------------------------------------------------------------
    
    def create_real_session(self):
        pass # Real robot session creation not implemented in this example.
    
    def execute_state_on_real_robot(self, state):
        pass  # Real robot execution not implemented in this example.

    # -- Virtual robot ---------------------------------------------------------

    def execute_state_on_virtual_robot(self, virtual_session, state, primitive_name: str = ""):
        """Execute state on virtual Gen3 Lite robot.

        Args:
            virtual_session: Virtual session for visualisation.
            state: Joint state array (48-dimensional internal format).
            primitive_name: Currently unused; kept for interface parity.
        """
        # Extract 7-DOF joint positions from internal 48-dim format
        joint_positions = state[:7]
        
        # Update virtual session configuration (similar to Pepper's set_cfg_array)
        virtual_session.set_cfg_array(joint_positions)

    # -- Real time handling ----------------------------------------------------

    def prepare_handle_rt(self):
        """Prepare for real-time head-tracking.

        Initializes any necessary state for the handle_rt() method.
        """
        self.rt_goal_indices = [
            self.get_joint_index("joint_1"),  # Base rotation for horizontal tracking
        ]

        # Real-time head-tracking state (simulated human target)
        angle = np.radians(-45)
        self._rt_targets = [
            0.0,
            angle/2,
            angle,
            angle/2,
            0.0,
            -angle/2,
            -angle,
            -angle/2,
        ]
        self._rt_hold_steps = 40   # 2 s at 20 Hz (core RT loop rate)
        self._rt_step = 0
        self._rt_index = 0
    
    def handle_rt(self, block):
        """Orient the base (joint_1) toward a cycling simulated target.

        Directly modifies the block's DMP goal for joint_1 for cycling targets.
        """

        target_angle = self._rt_targets[self._rt_index]

        goal = block.dmp.goal.copy()
        goal[self.rt_goal_indices[0]] = target_angle
        block.dmp.set_principle_parameters(p_goal=goal)

        # Advance the cycling target
        self._rt_step += 1
        if self._rt_step >= self._rt_hold_steps:
            self._rt_step = 0
            self._rt_index = (self._rt_index + 1) % len(self._rt_targets)

    
def create_robot(robot_name):
    """Factory function to create a Gen3 Lite robot instance.

    Args:
        robot_name: Name of the robot (should be ``'gen3_lite'``).

    Returns:
        Configured :class:`Gen3LiteRobot` instance.
    """
    return Gen3LiteRobot(robot_dir=f"robot_{robot_name}")

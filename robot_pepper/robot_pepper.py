"""
Pepper Robot Class

This module provides a robot-specific implementation for the Pepper humanoid robot.
It encapsulates all Pepper-specific functionality including trajectory translation,
joint mapping, and robot configuration.
"""

import sys
import numpy as np
from pathlib import Path

# Add parent directory to path to import robot_base
sys.path.insert(0, str(Path(__file__).parent.parent))
from session import RobotBase


class PepperRobot(RobotBase):
    """Pepper humanoid robot implementation.

    Handles Pepper-specific trajectory translation, joint mapping,
    and configuration management.
    """

    # -- Definitions -----------------------------------------------------------
    
    # Pepper's default standing joint angles (in degrees)
    STAND_INIT_JOINT_ANGLES = {
        "HeadPitch": -10.0,
        "HipPitch": -2.0,
        "HipRoll": 0.0,
        "KneePitch": 0.0,
        "LElbowRoll": -30.0,
        "LElbowYaw": -70.0,
        "LHand": 0.0,
        "LShoulderPitch": 90.0,
        "LShoulderRoll": 10.0,
        "LWristYaw": 0.0,
        "RElbowRoll": 30.0,
        "RElbowYaw": 70.0,
        "RHand": 0.0,
        "RShoulderPitch": 90.0,
        "RShoulderRoll": -10.0,
        "RWristYaw": 0.0,
        "HeadYaw": 0.0
    }
    
    # Mapping from joint names to indices in the internal representation
    JOINT_NAME_TO_IDX = {
        "KneePitch": 0,
        "HipPitch": 1,
        "HipRoll": 2,
        "HeadYaw": 3,
        "HeadPitch": 4,
        "LShoulderPitch": 5,
        "LShoulderRoll": 6,
        "LElbowYaw": 7,
        "LElbowRoll": 8,
        "LWristYaw": 9,
        "LHand": 10,
        "RShoulderPitch": 11,
        "RShoulderRoll": 12,
        "RElbowYaw": 13,
        "RElbowRoll": 14,
        "RWristYaw": 15,
        "RHand": 16
    }
    
    def __init__(self, robot_dir="robot_pepper"):
        """Initialize Pepper robot.

        Args:
            robot_dir: Path to the Pepper robot configuration directory.
        """
        super().__init__(robot_dir)
        # URDF path can be overridden, but defaults to description folder inside robot directory
        self.urdf_path = self.config.get(
            'urdf_path',
            str(self.robot_dir / 'robot_pepper_description' / 'robot_pepper.urdf')
        )

        self._ensure_follow_through_data()
        self.prepare_handle_rt()  # Prepare real-time head-tracking state        
    
    # -- Robot description related ---------------------------------------------
    
    def get_joint_names(self):
        """Return ordered list of Pepper joint names."""
        # Sort by index to get ordered list
        sorted_joints = sorted(self.JOINT_NAME_TO_IDX.items(), key=lambda x: x[1])
        return [name for name, idx in sorted_joints]
    
    def get_joint_index(self, joint_name):
        """Return the index for the given joint name.

        Args:
            joint_name: Name of the joint.

        Returns:
            Index of the joint in the internal representation.

        Raises:
            ValueError: If *joint_name* is unknown.
        """
        if joint_name not in self.JOINT_NAME_TO_IDX:
            raise ValueError(f"Unknown joint name: {joint_name}")
        return self.JOINT_NAME_TO_IDX[joint_name]
    
    def get_initial_joint_angles(self):
        """Return standing-init joint angles in radians for the virtual session."""
        return {k: np.deg2rad(v) for k, v in self.STAND_INIT_JOINT_ANGLES.items()}    

    def get_urdf_path(self):
        """Get path to Pepper's URDF file."""
        return self.urdf_path
    
    # -- Real robot ------------------------------------------------------------
    
    def create_real_session(self):
        pass # Real robot session creation not implemented in this example.
    
    def execute_state_on_real_robot(self, state):
        pass  # Real robot execution not implemented in this example.

    # -- Virtual robot ---------------------------------------------------------
    
    def execute_state_on_virtual_robot(self, virtual_session, state, primitive_name: str = ""):
        """Execute state on virtual Pepper robot.

        Converts from the 48-dim internal format to Pepper's virtual
        joint representation and pushes it to the viewer.

        Args:
            virtual_session: Virtual session for visualisation.
            state: Joint state array (48-dimensional internal format).
            primitive_name: Currently unused; kept for interface parity.
        """
        # Convert from 48-dim internal format to Pepper's virtual representation
        converted_state = self.convert_joint_array_for_virtual(state)
        virtual_session.set_cfg_array(converted_state)
    
    # -- Real time handling ----------------------------------------------------
    
    def prepare_handle_rt(self):
        """Prepare for real-time head-tracking.

        Initializes any necessary state for the handle_rt() method.
        """
        self.rt_goal_indices = [
            self.get_joint_index("HeadYaw"),
            self.get_joint_index("HeadPitch")
        ]

        # Real-time head-tracking state (simulated human target)
        yaw = np.radians(30)
        pitch = np.radians(10)
        self._rt_targets = [
            ( yaw,  -pitch),  # bottom-right
            ( yaw,   pitch),  # top-right
            ( yaw/2, pitch),
            ( 0,     pitch),
            ( 0,    -pitch),
            (-yaw/2,-pitch),
            (-yaw,  -pitch),  # bottom-left
            (-yaw,   pitch),  # top-left
            ( 0,     pitch),
        ]
        self._rt_hold_steps = 40   # 2 s at 20 Hz (core RT loop rate)
        self._rt_step = 0
        self._rt_index = 0
    
    def handle_rt(self, block):
        """Head-tracking toward a cycling simulated target.

        Directly modifies the block's DMP goal for HeadYaw / HeadPitch.
        """
        target_yaw, target_pitch = self._rt_targets[self._rt_index]

        state = block.dmp.get_state()
        goal = block.dmp.goal.copy()

        gain = 0.01
        goal[self.rt_goal_indices[0]] += gain * (target_yaw - state["y"][self.rt_goal_indices[0]])
        goal[self.rt_goal_indices[1]] += gain * (target_pitch - state["y"][self.rt_goal_indices[1]])
        block.dmp.set_principle_parameters(p_goal=goal)

        # Advance the cycling target
        self._rt_step += 1
        if self._rt_step >= self._rt_hold_steps:
            self._rt_step = 0
            self._rt_index = (self._rt_index + 1) % len(self._rt_targets)

    # -- Helper functions ------------------------------------------------------   

    @staticmethod
    def convert_joint_array_for_virtual(input_array):
        """Convert 48-dim internal joint array to Pepper's virtual representation.

        Maps from the internal representation to the specific joints used
        by Pepper's virtual visualisation.

        Args:
            input_array: 48-dimensional joint state array.

        Returns:
            Array with joints ordered for Pepper's virtual session.
        """
        joint_names = [
            "KneePitch", "HipPitch", "HipRoll", "HeadYaw", "HeadPitch",
            "LShoulderPitch", "LShoulderRoll", "LElbowYaw", "LElbowRoll", "LWristYaw", "LHand",
            "LFinger21", "LFinger22", "LFinger23", "LFinger11", "LFinger12", "LFinger13",
            "LFinger41", "LFinger42", "LFinger43", "LFinger31", "LFinger32", "LFinger33",
            "LThumb1", "LThumb2", "RShoulderPitch", "RShoulderRoll", "RElbowYaw", "RElbowRoll",
            "RWristYaw", "RHand", "RFinger41", "RFinger42", "RFinger43", "RFinger31",
            "RFinger32", "RFinger33", "RFinger21", "RFinger22", "RFinger23", "RFinger11",
            "RFinger12", "RFinger13", "RThumb1", "RThumb2", "WheelFL", "WheelB", "WheelFR"
        ]

        target_joints = [
            "HeadYaw", "HeadPitch", "HipRoll", "HipPitch", "KneePitch",
            "LShoulderPitch", "LShoulderRoll", "LElbowYaw", "LElbowRoll", "LWristYaw", "LHand",
            "RShoulderPitch", "RShoulderRoll", "RElbowYaw", "RElbowRoll", "RWristYaw", "RHand",
            "WheelFL", "WheelB", "WheelFR"
        ]

        if len(input_array) != 48:
            raise ValueError("Input array must have 48 elements.")

        target_indices = [joint_names.index(joint) for joint in target_joints]
        return input_array[target_indices] 



def create_robot(robot_name):
    """Factory function to create a Pepper robot instance.

    Args:
        robot_name: Name of the robot (should be ``'pepper'``).

    Returns:
        Configured :class:`PepperRobot` instance.
    """
    return PepperRobot(robot_dir=f"robot_{robot_name}")

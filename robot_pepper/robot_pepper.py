"""
Pepper Robot Class

This module provides a robot-specific implementation for the Pepper humanoid robot.
It encapsulates all Pepper-specific functionality including trajectory translation,
joint mapping, and robot configuration.
"""

import os
import sys
import yaml
import numpy as np
from pathlib import Path

# Add parent directory to path to import robot_base
sys.path.insert(0, str(Path(__file__).parent.parent))
from session import RobotBase


class PepperRobot(RobotBase):
    """
    Pepper humanoid robot implementation.
    
    Handles Pepper-specific trajectory translation, joint mapping,
    and configuration management.
    """
    
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
        """
        Initialize Pepper robot.
        
        Args:
            robot_dir: Path to the Pepper robot configuration directory
        """
        super().__init__(robot_dir)
        # URDF path can be overridden, but defaults to description folder inside robot directory
        self.urdf_path = self.config.get(
            'urdf_path',
            str(self.robot_dir / 'robot_pepper_description' / 'robot_pepper.urdf')
        )

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

    def translate_trajectory_to_internal(self, pepper_traj, use_zero=False):
        """
        Translate Pepper trajectory format to internal EDMP format.
        
        Args:
            pepper_traj: Dictionary with joint names as keys and position arrays as values
            use_zero: If True, initialize with zeros; if False, use standing pose
            
        Returns:
            numpy array of shape (timesteps, num_joints) in internal format
        """
        # Get the number of timesteps from the first trajectory
        first_key = next(iter(pepper_traj.keys()))
        num_timesteps = len(pepper_traj[first_key])
        
        # Initialize with zeros or standing pose
        # Using 48 as total joint dimension (expandable for future use)
        edmp_traj = np.zeros((num_timesteps, 48))
        
        # Iterate through time steps
        for i in range(num_timesteps):
            # If not using zero, set standing init values
            if not use_zero:
                for joint_name, angle_deg in self.STAND_INIT_JOINT_ANGLES.items():
                    if joint_name in self.JOINT_NAME_TO_IDX:
                        idx = self.JOINT_NAME_TO_IDX[joint_name]
                        edmp_traj[i, idx] = np.deg2rad(angle_deg)
            
            # Overwrite with trajectory data
            for joint_name, positions in pepper_traj.items():
                # Skip empty or unnamed columns
                if joint_name == "Unnamed: 0" or len(joint_name) <= 3:
                    continue
                
                if joint_name in self.JOINT_NAME_TO_IDX:
                    idx = self.JOINT_NAME_TO_IDX[joint_name]
                    edmp_traj[i, idx] = positions[i]
        
        return edmp_traj
    
    def translate_trajectory_to_external(self, internal_traj):
        """
        Translate internal EDMP format to Pepper trajectory format.
        
        Args:
            internal_traj: numpy array of shape (timesteps, num_joints)
            
        Returns:
            Dictionary with joint names as keys and position arrays as values
        """
        pepper_traj = {}
        
        # Create reverse mapping
        idx_to_joint = {idx: name for name, idx in self.JOINT_NAME_TO_IDX.items()}
        
        # Extract each joint's trajectory
        for idx, joint_name in idx_to_joint.items():
            if idx < internal_traj.shape[1]:  # Make sure index is valid
                pepper_traj[joint_name] = internal_traj[:, idx].tolist()
        
        return pepper_traj
    
    def get_joint_names(self):
        """
        Get ordered list of Pepper joint names.
        
        Returns:
            List of joint names in index order
        """
        # Sort by index to get ordered list
        sorted_joints = sorted(self.JOINT_NAME_TO_IDX.items(), key=lambda x: x[1])
        return [name for name, idx in sorted_joints]
    
    def get_joint_index(self, joint_name):
        """
        Get the index for a specific joint name.
        
        Args:
            joint_name: Name of the joint
            
        Returns:
            Index of the joint in the internal representation
        """
        if joint_name not in self.JOINT_NAME_TO_IDX:
            raise ValueError(f"Unknown joint name: {joint_name}")
        return self.JOINT_NAME_TO_IDX[joint_name]
    
    def get_initial_joint_angles(self):
        """Return standing-init joint angles in radians for the virtual session."""
        return {k: np.deg2rad(v) for k, v in self.STAND_INIT_JOINT_ANGLES.items()}

    def handle_rt(self, block):
        """Head-tracking toward a cycling simulated target.

        Directly modifies the block's DMP goal for HeadYaw / HeadPitch.
        """
        target_yaw, target_pitch = self._rt_targets[self._rt_index]

        yaw_idx = self.JOINT_NAME_TO_IDX["HeadYaw"]
        pitch_idx = self.JOINT_NAME_TO_IDX["HeadPitch"]

        state = block.dmp.get_state()
        goal = block.dmp.goal.copy()

        gain = 0.01
        goal[yaw_idx] += gain * (target_yaw - state["y"][yaw_idx])
        goal[pitch_idx] += gain * (target_pitch - state["y"][pitch_idx])
        block.dmp.set_principle_parameters(p_goal=goal)

        # Advance the cycling target
        self._rt_step += 1
        if self._rt_step >= self._rt_hold_steps:
            self._rt_step = 0
            self._rt_index = (self._rt_index + 1) % len(self._rt_targets)

    def get_urdf_path(self):
        """Get path to Pepper's URDF file."""
        return self.urdf_path
    
    def set_urdf_path(self, path):
        """Set custom URDF path."""
        self.urdf_path = path
    
    def prepare_real_robot_execution(self, session):
        """
        Prepare real robot for execution (Pepper-specific).
        
        Args:
            session: Pepper robot session
            
        Returns:
            Tuple of (names, limits_dict, mask) needed for execution
        """
        import peppertoolbox
        return peppertoolbox.prepare_pepper_execution(session.motion_service)
    
    def execute_state_on_real_robot(self, session, state, exec_data=None):
        """
        Execute state on real Pepper robot.
        
        Args:
            session: Pepper robot session
            state: Joint state array to execute
            exec_data: Tuple of (names, limits_dict, mask) from prepare_real_robot_execution
        """
        import peppertoolbox
        if exec_data is None:
            # If not provided, prepare it now
            exec_data = self.prepare_real_robot_execution(session)
        names, limits_dict, mask = exec_data
        peppertoolbox.execute_state(session, state, names, limits_dict, mask)
    
    def execute_state_on_virtual_robot(self, virtual_session, state):
        """
        Execute state on virtual Pepper robot.
        
        Args:
            virtual_session: Virtual session for visualization
            state: Joint state array to execute (48-dimensional internal format)
        """
        # Convert from 48-dim internal format to Pepper's virtual representation
        converted_state = self.convert_joint_array_for_virtual(state)
        virtual_session.set_cfg_array(converted_state)
    
    @staticmethod
    def convert_joint_array_for_virtual(input_array):
        """
        Convert 48-dimensional internal joint array to Pepper's virtual representation.
        
        This maps from the internal representation to the specific joints used
        by Pepper's virtual visualization.
        
        Args:
            input_array: 48-dimensional joint state array
            
        Returns:
            Array with joints ordered for Pepper's virtual session
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
    """
    Factory function to create a Pepper robot instance.
    
    Args:
        robot_name: Name of the robot (should be 'pepper')
        
    Returns:
        PepperRobot instance
    """
    return PepperRobot(robot_dir=f"robot_{robot_name}")


if __name__ == "__main__":
    # Example usage
    pepper = PepperRobot()
    
    print("Pepper Robot Configuration:")
    print(f"Capabilities: {pepper.get_capabilities()}")
    print(f"Number of primitives: {len(pepper.get_primitive_lib())}")
    print(f"Joint names: {pepper.get_joint_names()}")
    print(f"URDF path: {pepper.get_urdf_path()}")

"""
Kinova Gen3 Lite Robot Class

This module provides a robot-specific implementation for the Kinova Gen3 Lite robot arm.
It encapsulates all Gen3 Lite-specific functionality including trajectory translation,
joint mapping, and robot configuration.
"""

import os
import sys
import yaml
import numpy as np
from pathlib import Path
import logging

# Add parent directory to path to import robot_base
sys.path.insert(0, str(Path(__file__).parent.parent))
from session import RobotBase


class Gen3LiteRobot(RobotBase):
    """
    Kinova Gen3 Lite robot arm implementation.
    
    Handles Gen3 Lite-specific trajectory translation, joint mapping,
    and configuration management. This is a 7-DOF robot arm.
    """
    
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
        """
        Initialize Gen3 Lite robot.
        
        Args:
            robot_dir: Path to the Gen3 Lite robot configuration directory
        """
        super().__init__(robot_dir)
        # URDF path can be overridden, but defaults to description folder inside robot directory
        self.urdf_path = self.config.get(
            'urdf_path',
            str(self.robot_dir / 'robot_gen3_lite_description' / 'robot_ren3_lite.urdf')
        )

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
        
    def translate_trajectory_to_internal(self, external_traj, use_zero=False):
        """
        Translate Gen3 Lite trajectory format to internal EDMP format.
        
        Args:
            external_traj: numpy array of shape (timesteps, 7) with joint positions
            use_zero: If True, initialize with zeros; if False, use default pose
            
        Returns:
            numpy array of shape (timesteps, 48) in internal format
        """
        if isinstance(external_traj, dict):
            # If trajectory is a dictionary, convert to array
            num_timesteps = len(next(iter(external_traj.values())))
            traj_array = np.zeros((num_timesteps, 7))
            for joint_name, positions in external_traj.items():
                if joint_name in self.JOINT_NAME_TO_IDX:
                    idx = self.JOINT_NAME_TO_IDX[joint_name]
                    traj_array[:, idx] = positions
            external_traj = traj_array
        
        num_timesteps = external_traj.shape[0]
        
        # Initialize with zeros or default pose
        # Using 48 as total joint dimension (expandable for future use)
        edmp_traj = np.zeros((num_timesteps, 48))
        
        # Copy the 7-DOF trajectory to the first 7 dimensions
        for i in range(num_timesteps):
            if not use_zero:
                edmp_traj[i, :7] = self.DEFAULT_JOINT_ANGLES
            
            # Overwrite with actual trajectory data
            edmp_traj[i, :7] = external_traj[i, :]
        
        return edmp_traj
    
    def translate_trajectory_to_external(self, internal_traj):
        """
        Translate internal EDMP format to Gen3 Lite trajectory format.
        
        Args:
            internal_traj: numpy array of shape (timesteps, 48)
            
        Returns:
            numpy array of shape (timesteps, 7) with joint positions
        """
        # Extract the first 7 dimensions (Gen3 Lite joints)
        return internal_traj[:, :7]
    
    def get_joint_names(self):
        """
        Get ordered list of Gen3 Lite joint names.
        
        Returns:
            List of joint names in order
        """
        return self.JOINT_NAMES.copy()
    
    def get_joint_index(self, joint_name):
        """
        Get the index for a specific joint name.
        
        Args:
            joint_name: Name of the joint (e.g., "joint_1")
            
        Returns:
            Index of the joint
        """
        if joint_name not in self.JOINT_NAME_TO_IDX:
            raise ValueError(f"Unknown joint name: {joint_name}")
        return self.JOINT_NAME_TO_IDX[joint_name]
    
    def get_initial_joint_angles(self):
        """Return default joint angles in radians for the virtual session."""
        return {name: float(self.DEFAULT_JOINT_ANGLES[i]) for i, name in enumerate(self.JOINT_NAMES)}
    
    def handle_rt(self, block):
        """Orient the base (joint_1) toward a cycling simulated target.

        Directly modifies the block's DMP goal for joint_1 for cycling targets.
        """
        joint_idx = self.JOINT_NAME_TO_IDX["joint_1"]

        target_angle = self._rt_targets[self._rt_index]

        goal = block.dmp.goal.copy()
        goal[joint_idx] = target_angle
        block.dmp.set_principle_parameters(p_goal=goal)

        # Advance the cycling target
        self._rt_step += 1
        if self._rt_step >= self._rt_hold_steps:
            self._rt_step = 0
            self._rt_index = (self._rt_index + 1) % len(self._rt_targets)

    def get_urdf_path(self):
        """Get path to URDF file."""
        return self.urdf_path
    
    def set_urdf_path(self, path):
        """Set custom URDF path."""
        self.urdf_path = path
    
    def prepare_real_robot_execution(self, session):
        """
        Prepare real robot for execution (Gen3 Lite-specific).
        
        Args:
            session: Gen3 Lite robot session
            
        Returns:
            None or preparation data
        """
        # TODO: Implement when real robot interface is available
        return None
    
    def execute_state_on_real_robot(self, session, state, exec_data=None):
        """
        Execute state on real Gen3 Lite robot.
        
        Args:
            session: Gen3 Lite robot session
            state: Joint state array to execute (first 7 dimensions)
            exec_data: Optional execution data
        """
        # TODO: Implement real robot execution when hardware interface is ready
        # Extract 7-DOF joint positions from state
        joint_positions = state[:7]
        raise NotImplementedError("Real robot execution not yet implemented for Gen3 Lite")
    
    def execute_state_on_virtual_robot(self, virtual_session, state):
        """
        Execute state on virtual Gen3 Lite robot.
        
        Args:
            virtual_session: Virtual session for visualization
            state: Joint state array to execute (48-dimensional internal format)
        """
        # Extract 7-DOF joint positions from internal 48-dim format
        joint_positions = state[:7]
        
        # Update virtual session configuration (similar to Pepper's set_cfg_array)
        virtual_session.set_cfg_array(joint_positions)


def create_robot(robot_name):
    """
    Factory function to create a Gen3 Lite robot instance.
    
    Args:
        robot_name: Name of the robot (should be 'gen3_lite')
        
    Returns:
        Gen3LiteRobot instance
    """
    return Gen3LiteRobot(robot_dir=f"robot_{robot_name}")


if __name__ == "__main__":
    # Example usage
    gen3_lite = Gen3LiteRobot()
    
    print("Gen3 Lite Robot Configuration:")
    print(f"Capabilities: {gen3_lite.get_capabilities()}")
    print(f"Number of primitives: {len(gen3_lite.get_primitive_lib())}")
    print(f"Joint names: {gen3_lite.get_joint_names()}")
    print(f"URDF path: {gen3_lite.get_urdf_path()}")
    print(f"Primitive path: {gen3_lite.get_primitive_path()}")

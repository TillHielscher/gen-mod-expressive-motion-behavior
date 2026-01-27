"""
Unitree Go2 Robot Class

This module provides a robot-specific implementation for the Unitree Go2 quadruped robot.
It encapsulates all Go2-specific functionality including trajectory translation,
joint mapping, and robot configuration.
"""

import os
import sys
import yaml
import numpy as np
from pathlib import Path

# Add parent directory to path to import robot_base
sys.path.insert(0, str(Path(__file__).parent.parent))
from robot_base import RobotBase


class Go2Robot(RobotBase):
    """
    Unitree Go2 quadruped robot implementation.
    
    Handles Go2-specific trajectory translation, joint mapping,
    and configuration management. This is a 12-DOF quadruped (4 legs × 3 joints).
    """
    
    # Go2's default standing joint angles (in radians)
    # Format: [FL_hip, FL_thigh, FL_calf, FR_hip, FR_thigh, FR_calf, 
    #          RL_hip, RL_thigh, RL_calf, RR_hip, RR_thigh, RR_calf]
    DEFAULT_JOINT_ANGLES = np.array([
        0.0, 0.78, -1.40,  # Front Left
        0.0, 0.78, -1.41,  # Front Right
        0.0, 0.81, -1.45,  # Rear Left
        0.09, 0.78, -1.39  # Rear Right
    ])
    
    # Joint names for the 12-DOF quadruped
    JOINT_NAMES = [
        "FL_hip_joint",   # Front Left hip
        "FL_thigh_joint", # Front Left thigh
        "FL_calf_joint",  # Front Left calf
        "FR_hip_joint",   # Front Right hip
        "FR_thigh_joint", # Front Right thigh
        "FR_calf_joint",  # Front Right calf
        "RL_hip_joint",   # Rear Left hip
        "RL_thigh_joint", # Rear Left thigh
        "RL_calf_joint",  # Rear Left calf
        "RR_hip_joint",   # Rear Right hip
        "RR_thigh_joint", # Rear Right thigh
        "RR_calf_joint",  # Rear Right calf
    ]
    
    # Mapping from joint names to indices
    JOINT_NAME_TO_IDX = {name: idx for idx, name in enumerate(JOINT_NAMES)}
    
    def __init__(self, robot_dir="robot_go2"):
        """
        Initialize Go2 robot.
        
        Args:
            robot_dir: Path to the Go2 robot configuration directory
        """
        super().__init__(robot_dir)
        # URDF path can be overridden, but defaults to description folder inside robot directory
        self.urdf_path = self.config.get(
            'urdf_path',
            str(self.robot_dir / 'robot_go2_description' / 'go2_description.urdf')
        )
        
    def translate_trajectory_to_internal(self, external_traj, use_zero=False):
        """
        Translate Go2 trajectory format to internal EDMP format.
        
        Args:
            external_traj: numpy array of shape (timesteps, 12) with joint positions
                          or dictionary with joint names as keys
            use_zero: If True, initialize with zeros; if False, use default pose
            
        Returns:
            numpy array of shape (timesteps, 48) in internal format
        """
        if isinstance(external_traj, dict):
            # If trajectory is a dictionary, convert to array
            num_timesteps = len(next(iter(external_traj.values())))
            traj_array = np.zeros((num_timesteps, 12))
            for joint_name, positions in external_traj.items():
                if joint_name in self.JOINT_NAME_TO_IDX:
                    idx = self.JOINT_NAME_TO_IDX[joint_name]
                    traj_array[:, idx] = positions
            external_traj = traj_array
        
        num_timesteps = external_traj.shape[0]
        
        # Initialize with zeros or default pose
        # Using 48 as total joint dimension (expandable for future use)
        edmp_traj = np.zeros((num_timesteps, 48))
        
        # Copy the 12-DOF trajectory to the first 12 dimensions
        for i in range(num_timesteps):
            if not use_zero:
                edmp_traj[i, :12] = self.DEFAULT_JOINT_ANGLES
            
            # Overwrite with actual trajectory data
            edmp_traj[i, :12] = external_traj[i, :]
        
        return edmp_traj
    
    def translate_trajectory_to_external(self, internal_traj):
        """
        Translate internal EDMP format to Go2 trajectory format.
        
        Args:
            internal_traj: numpy array of shape (timesteps, 48)
            
        Returns:
            numpy array of shape (timesteps, 12) with joint positions
        """
        # Extract the first 12 dimensions (Go2 joints)
        return internal_traj[:, :12]
    
    def get_joint_names(self):
        """
        Get ordered list of Go2 joint names.
        
        Returns:
            List of joint names in order
        """
        return self.JOINT_NAMES.copy()
    
    def get_joint_index(self, joint_name):
        """
        Get the index for a specific joint name.
        
        Args:
            joint_name: Name of the joint
            
        Returns:
            Index of the joint
        """
        if joint_name not in self.JOINT_NAME_TO_IDX:
            raise ValueError(f"Unknown joint name: {joint_name}")
        return self.JOINT_NAME_TO_IDX[joint_name]
    
    def get_urdf_path(self):
        """Get path to URDF file."""
        return self.urdf_path
    
    def set_urdf_path(self, path):
        """Set custom URDF path."""
        self.urdf_path = path
    
    def prepare_real_robot_execution(self, session):
        """
        Prepare real robot for execution (Go2-specific).
        
        Args:
            session: Go2 robot session
            
        Returns:
            None or preparation data
        """
        # TODO: Implement when real robot interface is available
        return None
    
    def execute_state_on_real_robot(self, session, state, exec_data=None):
        """
        Execute state on real Go2 robot.
        
        Args:
            session: Go2 robot session
            state: Joint state array to execute (first 12 dimensions)
            exec_data: Optional execution data
        """
        # TODO: Implement real robot execution when hardware interface is ready
        # Extract 12-DOF joint positions from state
        joint_positions = state[:12]
        raise NotImplementedError("Real robot execution not yet implemented for Go2")
    
    def execute_state_on_virtual_robot(self, virtual_session, state):
        """
        Execute state on virtual Go2 robot.
        
        Args:
            virtual_session: Virtual session for visualization
            state: Joint state array to execute (48-dimensional internal format)
        """
        # Extract 12-DOF joint positions from internal 48-dim format
        joint_positions = state[:12]
        
        # Update virtual session configuration
        virtual_session.set_cfg_array(joint_positions)


def create_robot(robot_name):
    """
    Factory function to create a Go2 robot instance.
    
    Args:
        robot_name: Name of the robot (should be 'go2')
        
    Returns:
        Go2Robot instance
    """
    return Go2Robot(robot_dir=f"robot_{robot_name}")


if __name__ == "__main__":
    # Example usage
    go2 = Go2Robot()
    
    print("Go2 Robot Configuration:")
    print(f"Capabilities: {go2.get_capabilities()}")
    print(f"Number of primitives: {len(go2.get_primitive_lib())}")
    print(f"Joint names: {go2.get_joint_names()}")
    print(f"URDF path: {go2.get_urdf_path()}")
    print(f"Primitive path: {go2.get_primitive_path()}")

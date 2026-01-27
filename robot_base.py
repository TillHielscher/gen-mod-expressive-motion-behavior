"""
Base Robot Class Module

This module provides the abstract base class that all robot implementations
must inherit from. Place robot-specific implementations in their respective
robot_* directories (e.g., robot_pepper/robot_pepper.py).

This file can be imported by the main LLAG system to ensure a consistent
interface across all robot types.
"""

from abc import ABC, abstractmethod
from pathlib import Path
import yaml


class RobotBase(ABC):
    """
    Abstract base class for robot implementations.
    
    This defines the interface that all robot classes must implement,
    allowing the LLAG system to work with different robots without
    modification to the core code.
    """
    
    def __init__(self, robot_dir):
        """
        Initialize the robot with its configuration directory.
        
        Args:
            robot_dir: Path to the robot's configuration directory
        """
        self.robot_dir = Path(robot_dir)
        self.config = self._load_config()
        
    def _load_config(self):
        """Load the robot configuration from YAML file."""
        config_path = self.robot_dir / f"{self.robot_dir.name}.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Robot config not found at {config_path}")
        
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    @abstractmethod
    def translate_trajectory_to_internal(self, external_traj):
        """
        Translate from robot-specific format to internal EDMP format.
        
        Args:
            external_traj: Trajectory in robot-specific format
            
        Returns:
            Trajectory in internal EDMP format (numpy array)
        """
        pass
    
    @abstractmethod
    def translate_trajectory_to_external(self, internal_traj):
        """
        Translate from internal EDMP format to robot-specific format.
        
        Args:
            internal_traj: Trajectory in internal EDMP format
            
        Returns:
            Trajectory in robot-specific format
        """
        pass
    
    @abstractmethod
    def get_joint_names(self):
        """
        Get the ordered list of joint names for this robot.
        
        Returns:
            List of joint name strings
        """
        pass
    
    @abstractmethod
    def execute_state_on_real_robot(self, session, state, *args, **kwargs):
        """
        Execute state on real robot hardware.
        
        Args:
            session: Robot session/connection
            state: Joint state array to execute
            *args, **kwargs: Robot-specific additional parameters
        """
        pass
    
    @abstractmethod
    def execute_state_on_virtual_robot(self, virtual_session, state):
        """
        Execute state on virtual robot (visualization).
        
        Args:
            virtual_session: Virtual session for visualization
            state: Joint state array to execute
        """
        pass
    
    def prepare_real_robot_execution(self, session):
        """
        Prepare real robot for execution (optional, robot-specific).
        
        Args:
            session: Robot session
            
        Returns:
            Any preparation data needed for execution (or None)
        """
        return None
    
    def get_capabilities(self):
        """Get robot capabilities description."""
        return self.config.get("capabilities", "")
    
    def get_primitive_lib(self):
        """Get available motion primitives."""
        return self.config.get("primitive_lib", {})
    
    def get_idle_lib(self):
        """Get available idle motions."""
        return self.config.get("idle_lib", [])
    
    def get_parameter_ranges(self):
        """Get parameter ranges for motion modulation."""
        return self.config.get("parameter_ranges", {})
    
    def get_primitive_path(self):
        """Get path to saved primitives."""
        return str(self.robot_dir / "primitive_saved")
    
    def get_urdf_path(self):
        """Get path to URDF file for this robot."""
        # This should be overridden by specific robot implementations
        return None
    
    def get_robot_description_path(self):
        """Get path to robot description YAML file."""
        return str(self.robot_dir / f"{self.robot_dir.name}.yaml")

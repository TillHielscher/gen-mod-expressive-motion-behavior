"""
Robot-agnostic virtual session for loading URDF, visualizing in Viser, and controlling joints.

This module provides a generic VirtualSession class that can work with any robot URDF file,
replacing the Pepper-specific implementation from pepper-toolbox-main.
"""

import time
from typing import Dict, List, Optional, Union
from pathlib import Path

import numpy as np
from yourdfpy import URDF

import viser
from viser.extras import ViserUrdf


class VirtualSession:
    """
    A robot-agnostic virtual session for URDF visualization and joint control using Viser.
    
    This class loads a robot URDF, creates a Viser server for visualization, and provides
    methods to control joint positions either through GUI sliders or programmatically.
    
    Attributes:
        server: The Viser server instance for visualization
        viser_urdf: The ViserUrdf instance managing the robot visualization
        slider_handles: List of GUI slider handles for joint control
        initial_config: Initial joint configuration values
    """

    stand_init_joint_anges = {"HeadPitch": -10.0, "HipPitch": -2.0, "HipRoll": 0.0, "KneePitch": 0.0, "LElbowRoll": -30.0, "LElbowYaw": -70.0, "LHand": 0.0, "LShoulderPitch": 90.0, "LShoulderRoll": 10.0, "LWristYaw": 0.0, "RElbowRoll": 30.0, "RElbowYaw": 70.0, "RHand": 0.0, "RShoulderPitch": 90.0, "RShoulderRoll": -10.0, "RWristYaw": 0.0, "HeadYaw": 0.0}

    
    def __init__(
        self,
        urdf_path: Union[str, Path],
        initial_joint_angles: Optional[Dict[str, float]] = None,
        load_meshes: bool = True,
        load_collision_meshes: bool = False,
        port: int = 8088,
        grid_size: float = 2000.0,
    ):
        """
        Initialize a virtual session for a robot.
        
        Args:
            urdf_path: Path to the URDF file to load
            initial_joint_angles: Optional dict of joint names to initial angles (in radians)
            load_meshes: Whether to load visual meshes
            load_collision_meshes: Whether to load collision meshes
            port: Port number for the Viser server
            grid_size: Size of the ground grid in the visualization (width and height)
        """
        self.urdf_path = Path(urdf_path)
        if not self.urdf_path.exists():
            raise FileNotFoundError(f"URDF file not found: {self.urdf_path}")
        
        # Use stand_init_joint_anges as default if no initial_joint_angles provided
        if initial_joint_angles is None:
            initial_joint_angles = {k: np.deg2rad(v) for k, v in self.stand_init_joint_anges.items()}
        
        self.server = viser.ViserServer(port=port)
        self.load_meshes = load_meshes
        self.load_collision_meshes = load_collision_meshes
        
        # Load URDF
        yourdf = URDF.load(str(self.urdf_path))
        
        self.viser_urdf = ViserUrdf(
            self.server,
            urdf_or_path=yourdf,
            load_meshes=load_meshes,
            load_collision_meshes=load_collision_meshes,
            collision_mesh_color_override=(1.0, 0.0, 0.0, 0.5),
        )
        
        # Create sliders for joint control
        with self.server.gui.add_folder("Joint position control"):
            self.slider_handles, self.initial_config = self._create_robot_control_sliders(
                initial_joint_angles
            )
        
        # Add visibility checkboxes
        with self.server.gui.add_folder("Visibility"):
            self.show_meshes_cb = self.server.gui.add_checkbox(
                "Show meshes", self.viser_urdf.show_visual
            )
            self.show_collision_meshes_cb = self.server.gui.add_checkbox(
                "Show collision meshes", self.viser_urdf.show_collision
            )
        
        @self.show_meshes_cb.on_update
        def _(_):
            self.viser_urdf.show_visual = self.show_meshes_cb.value
        
        @self.show_collision_meshes_cb.on_update
        def _(_):
            self.viser_urdf.show_collision = self.show_collision_meshes_cb.value
        
        self.show_meshes_cb.visible = load_meshes
        self.show_collision_meshes_cb.visible = load_collision_meshes
        
        # Set initial robot configuration
        self.viser_urdf.update_cfg(np.array(self.initial_config))
        
        # Create grid
        trimesh_scene = self.viser_urdf._urdf.scene or self.viser_urdf._urdf.collision_scene
        self.server.scene.add_grid(
            "/grid",
            width=grid_size,
            height=grid_size,
            position=(
                0.0,
                0.0,
                trimesh_scene.bounds[0, 2] if trimesh_scene is not None else 0.0,
            ),
            cell_thickness=1000.0,
            shadow_opacity=1.0,
        )
        
        # Set up lighting
        self.server.scene.enable_default_lights()
        self.server.scene.configure_environment_map(None)
        self.server.scene.add_light_hemisphere(
            "/hemisphere_light", position=(10.0, 10.0, 0.0), intensity=0.5
        )
        self.server.scene.add_light_point(
            "/light_point",
            position=(5.0, -5.0, 0.0),
            distance=10.0,
            decay=0.0,
            intensity=0.5,
            cast_shadow=True,
            color=(255, 200, 220),
        )
        
        # Create joint reset button
        reset_button = self.server.gui.add_button("Reset")
        
        @reset_button.on_click
        def _(_):
            self.set_joint_states(self.initial_config)
    
    def _create_robot_control_sliders(
        self, initial_joint_angles: Optional[Dict[str, float]] = None
    ) -> tuple[List[viser.GuiInputHandle], List[float]]:
        """
        Create GUI sliders for controlling robot joints.
        
        Args:
            initial_joint_angles: Optional dict of joint names to initial angles (in radians)
        
        Returns:
            Tuple of (slider_handles, initial_config) where:
                - slider_handles: List of GUI slider handles
                - initial_config: List of initial joint position values
        """
        initial_joint_angles = initial_joint_angles or {}
        
        slider_handles: List[viser.GuiInputHandle] = []
        initial_config: List[float] = []
        
        for joint_name, (lower, upper) in self.viser_urdf.get_actuated_joint_limits().items():
            lower = lower if lower is not None else -np.pi
            upper = upper if upper is not None else np.pi
            
            # Use provided initial angle if available, otherwise use sensible default
            if joint_name in initial_joint_angles:
                initial_pos = np.clip(initial_joint_angles[joint_name], lower, upper)
            else:
                initial_pos = 0.0 if lower < -0.1 and upper > 0.1 else (lower + upper) / 2.0
            
            slider = self.server.gui.add_slider(
                label=joint_name,
                min=lower,
                max=upper,
                step=1e-3,
                initial_value=initial_pos,
            )
            
            slider.on_update(
                lambda _, sh=slider_handles: self.viser_urdf.update_cfg(
                    np.array([s.value for s in sh])
                )
            )
            
            slider_handles.append(slider)
            initial_config.append(initial_pos)
        
        return slider_handles, initial_config
    
    def get_joint_names(self) -> List[str]:
        """Get the list of actuated joint names in order."""
        return [s.label for s in self.slider_handles]
    
    def get_joint_limits(self) -> Dict[str, tuple[Optional[float], Optional[float]]]:
        """Get the joint limits for all actuated joints."""
        return self.viser_urdf.get_actuated_joint_limits()
    
    def get_current_joint_states(self) -> Dict[str, float]:
        """Get the current joint states as a dictionary."""
        return {s.label: s.value for s in self.slider_handles}
    
    def set_joint_states(self, joint_values: Union[List[float], Dict[str, float]]):
        """
        Set joint states by list (order matches sliders) or by dict (name: value in radians).
        
        Args:
            joint_values: Either a list of joint values (in order of sliders) or a dict
                         mapping joint names to values in radians
        """
        if isinstance(joint_values, dict):
            # Map dict to slider order
            values = []
            for s in self.slider_handles:
                values.append(joint_values.get(s.label, s.value))
            self.viser_urdf.update_cfg(np.array(values))
        else:
            self.viser_urdf.update_cfg(np.array(joint_values))
    
    def set_sliders_and_cfg(self, joint_dict: Dict[str, float]):
        """
        Update the GUI sliders for the given joint names and values,
        then update the robot configuration using all slider values.
        
        This method updates both the GUI and the visualization.
        
        Args:
            joint_dict: Dict mapping joint names to values in radians
        """
        for s in self.slider_handles:
            if s.label in joint_dict:
                s.value = joint_dict[s.label]
        self.viser_urdf.update_cfg(np.array([s.value for s in self.slider_handles]))
    
    def set_cfg(self, joint_dict: Dict[str, float]):
        """
        Efficiently update the robot configuration without updating GUI sliders.
        
        This is the preferred method for programmatic control as it avoids
        the overhead of updating GUI elements.
        
        Args:
            joint_dict: Dict mapping joint names to values in radians
        """
        # Build a NumPy array of current slider values
        values = np.array([s.value for s in self.slider_handles])
        # Update only the joints provided in joint_dict (do NOT update slider .value)
        for idx, s in enumerate(self.slider_handles):
            if s.label in joint_dict:
                values[idx] = joint_dict[s.label]
        self.viser_urdf.update_cfg(values)
    
    def set_cfg_array(self, values: Union[List[float], np.ndarray]):
        """
        Efficiently update the robot configuration using an array,
        without updating GUI sliders.
        
        Args:
            values: Array of joint values in radians, order matches sliders
        """
        self.viser_urdf.update_cfg(np.array(values))
    
    def wait_for_client(self, timeout: Optional[float] = None, poll_interval: float = 0.5):
        """
        Wait for at least one client to connect to the Viser server.
        
        Args:
            timeout: Maximum time to wait in seconds (None = wait forever)
            poll_interval: Time between connection checks in seconds
        
        Returns:
            True if a client connected, False if timeout was reached
        """
        start_time = time.time()
        while True:
            clients = self.server.get_clients()
            if clients:
                print(f"Connected clients: {len(clients)}")
                return True
            
            if timeout is not None and (time.time() - start_time) > timeout:
                return False
            
            print("Waiting for clients to connect...")
            time.sleep(poll_interval)
    
    def run_forever(self):
        """Keep the session running indefinitely."""
        while True:
            time.sleep(1)


def load_urdf_for_robot(robot_name: str, urdf_folder: str = ".") -> Path:
    """
    Helper function to find a URDF file for a specific robot.
    
    Args:
        robot_name: Name of the robot (e.g., 'pepper', 'go2', 'kinova', 'franka')
        urdf_folder: Base folder to search for URDF files (defaults to current directory)
    
    Returns:
        Path to the URDF file
    
    Raises:
        FileNotFoundError: If no URDF file is found for the robot
    """
    base_path = Path(urdf_folder)
    
    # Common URDF naming patterns
    search_patterns = [
        f"{robot_name}_urdf/{robot_name}.urdf",
        f"{robot_name}_urdf/{robot_name}_pruned.urdf",
        f"{robot_name}_urdf/{robot_name}_description.urdf",
        f"pepper-toolbox-main/src/peppertoolbox/urdf/{robot_name}_pruned.urdf",
        f"kinematic_analysis/{robot_name}_urdf/{robot_name}.urdf",
        f"kinematic_analysis/{robot_name}_urdf/{robot_name}_pruned.urdf",
    ]
    
    for pattern in search_patterns:
        urdf_path = base_path / pattern
        if urdf_path.exists():
            return urdf_path
    
    raise FileNotFoundError(
        f"Could not find URDF file for robot '{robot_name}'. "
        f"Searched in: {', '.join(search_patterns)}"
    )

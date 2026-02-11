"""
Unitree Go2 Robot Class

This module provides a robot-specific implementation for the Unitree Go2 quadruped robot.
It encapsulates all Go2-specific functionality including trajectory translation,
joint mapping, and robot configuration.
"""

import logging
import sys
import numpy as np
from pathlib import Path
from yourdfpy import URDF
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize

# Add parent directory to path to import robot_base
sys.path.insert(0, str(Path(__file__).parent.parent))
from session import RobotBase


class Go2Robot(RobotBase):
    """Unitree Go2 quadruped robot implementation.

    Handles Go2-specific trajectory translation, joint mapping,
    and configuration management.  This is a 12-DOF quadruped (4 legs × 3 joints).
    """

    # -- Definitions -----------------------------------------------------------
    
    # Go2's default standing joint angles (in radians)
    # Format: [FR_hip, FR_thigh, FR_calf, FL_hip, FL_thigh, FL_calf, 
    #          RR_hip, RR_thigh, RR_calf, RL_hip, RL_thigh, RL_calf]
    # Matches DMP ordering: FR_0-2, FL_0-2, RR_0-2, RL_0-2
    DEFAULT_JOINT_ANGLES = np.array([
        0.0, 0.78, -1.41,  # Front Right (FR)
        0.0, 0.78, -1.40,  # Front Left (FL)
        0.09, 0.78, -1.39, # Rear Right (RR)
        0.0, 0.81, -1.45   # Rear Left (RL)
    ])
    
    # Joint names for the 12-DOF quadruped
    # Order matches DMP: FR, FL, RR, RL
    JOINT_NAMES = [
        "FR_hip_joint",   # FR_0 -> 0
        "FR_thigh_joint", # FR_1 -> 1
        "FR_calf_joint",  # FR_2 -> 2
        "FL_hip_joint",   # FL_0 -> 3
        "FL_thigh_joint", # FL_1 -> 4
        "FL_calf_joint",  # FL_2 -> 5
        "RR_hip_joint",   # RR_0 -> 6
        "RR_thigh_joint", # RR_1 -> 7
        "RR_calf_joint",  # RR_2 -> 8
        "RL_hip_joint",   # RL_0 -> 9
        "RL_thigh_joint", # RL_1 -> 10
        "RL_calf_joint",  # RL_2 -> 11
    ]
    
    # Mapping from joint names to indices
    JOINT_NAME_TO_IDX = {name: idx for idx, name in enumerate(JOINT_NAMES)}

    # Manual height offset (metres) added to the optimised Z position.
    # Negative values push the robot closer to the ground, positive lifts it.
    GROUND_Z_OFFSET: float = -0.33

    # Primitives where specific feet are lifted and must be excluded from
    # the ground-contact optimisation.  Maps primitive name → list of URDF
    # foot link names to ignore.  Uses fuzzy substring matching (same as
    # TimelineBlock), so "shake_hand" matches "shake_hand_1", etc.
    RAISED_FEET: dict[str, list[str]] = {
        "shake_hand":  ["FR_foot"],
    }
    
    def __init__(self, robot_dir="robot_go2", enable_floating_base=True):
        """Initialize Go2 robot.

        Args:
            robot_dir: Path to the Go2 robot configuration directory.
            enable_floating_base: If True, compute base transforms to keep
                feet planted (default).  If False, use standard fixed-base
                visualisation.
        """
        super().__init__(robot_dir)
        # URDF path can be overridden, but defaults to description folder inside robot directory
        self.urdf_path = self.config.get(
            'urdf_path',
            str(self.robot_dir / 'robot_go2_description' / 'robot_go2.urdf')
        )

        self.prepare_handle_rt()        
        
        # Load URDF for FK computation (only if floating base enabled)
        self._urdf = None
        self._load_urdf()       
        
    # -- Robot description related ---------------------------------------------
    
    def get_joint_names(self):
        """Return ordered list of Go2 joint names."""
        return self.JOINT_NAMES.copy()
    
    def get_joint_index(self, joint_name):
        """Return the index for the given joint name.

        Args:
            joint_name: Name of the joint.

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
        """Execute state on virtual Go2 robot.

        Extracts the 12-DOF joints, reorders them for the URDF, updates the
        viewer, and optionally computes the floating-base transform.

        Args:
            virtual_session: Virtual session for visualisation.
            state: Joint state array (48-dimensional internal format).
            primitive_name: Name of the currently playing primitive (forwarded
                to :meth:`compute_base_transform` for raised-foot handling).
        """
        # Extract 12-DOF joint positions from internal format
        joint_positions = state[:12]
        
        # Convert from DMP order to URDF order
        converted_state = self.convert_joint_array_for_virtual(joint_positions)
        
        # Update virtual session with joint configuration
        virtual_session.set_cfg_array(converted_state)
        
        # Compute floating-base transform (feet planted on ground)
        base_transform = self.compute_base_transform(converted_state, primitive_name)
        if base_transform is not None:
            position, quaternion = base_transform
            virtual_session.set_base_transform(position, quaternion)
    
    # -- Real time handling ----------------------------------------------------

    def prepare_handle_rt(self):
        """Prepare for real-time head-tracking.

        Initializes any necessary state for the handle_rt() method.
        """
        self.rt_goal_indices = []  # No real-time adjustments for Go2 in this implementation

    def handle_rt(self, block):
        pass  # No real-time adjustments for Go2 in this implementation
    
    # -- Helper functions ------------------------------------------------------    

    def _load_urdf(self):
        """Load URDF for forward kinematics."""
        try:
            self._urdf = URDF.load(self.urdf_path)
            
        except Exception as e:
            print(f"Warning: Could not load URDF for FK computation: {e}")
            self._urdf = None
        
        # Foot link names in URDF
        self.foot_links = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]

    @staticmethod
    def convert_joint_array_for_virtual(input_array):
        """Convert 12-dim DMP joint array to Go2's URDF joint order.

        The DMP uses order FR, FL, RR, RL (indices 0-11).
        The URDF expects order FL, FR, RL, RR.

        Args:
            input_array: 12-dimensional joint state array in DMP order.

        Returns:
            Array with joints reordered for Go2's virtual session.
        """
        if len(input_array) != 12:
            raise ValueError(f"Input array must have 12 elements, got {len(input_array)}")
        
        # DMP order: FR(0-2), FL(3-5), RR(6-8), RL(9-11)
        # URDF order: FL(0-2), FR(3-5), RL(6-8), RR(9-11)
        # Mapping: FL from 3-5, FR from 0-2, RL from 9-11, RR from 6-8
        
        reordered = np.array([
            input_array[3], input_array[4], input_array[5],  # FL (hip, thigh, calf)
            input_array[0], input_array[1], input_array[2],  # FR (hip, thigh, calf)
            input_array[9], input_array[10], input_array[11], # RL (hip, thigh, calf)
            input_array[6], input_array[7], input_array[8],   # RR (hip, thigh, calf)
        ])
        
        return reordered
    
    def compute_foot_positions(self, joint_cfg_dict):
        """Compute the world positions of all four feet given joint configuration.

        Args:
            joint_cfg_dict: Dictionary mapping joint names to values (URDF order).

        Returns:
            Dictionary mapping foot link names to ``[x, y, z]`` positions,
            or ``None`` on failure.
        """
        if self._urdf is None:
            return None
        
        try:
            # Update URDF configuration
            self._urdf.update_cfg(configuration=joint_cfg_dict)
            
            # Get transforms for foot links
            foot_positions = {}
            for foot_link in self.foot_links:
                # Get 4x4 transform matrix for this link
                transform = self._urdf.get_transform(foot_link)
                # Extract translation (last column, first 3 rows)
                foot_positions[foot_link] = transform[:3, 3]
            
            return foot_positions
        except Exception as e:
            print(f"Warning: FK computation failed: {e}")
            return None    

    def compute_base_transform(self, joint_array, primitive_name: str = ""):
        """Compute base transform to keep ground-contact feet planted.

        Feet listed in ``RAISED_FEET`` for the current *primitive_name* are
        excluded from the optimisation so that primitives like *shake_hand*
        that intentionally lift a leg work correctly.

        Solves (over planted feet only):
            minimize  Σ_i ‖p_ground_i − (p_base + R_base · p_foot_i)‖²

        Args:
            joint_array: 12-dimensional array in URDF order.
            primitive_name: Name of the currently playing primitive (used to
                look up ``RAISED_FEET``).

        Returns:
            ``(position, quaternion)`` or *None* if FK is unavailable.
        """
        if self._urdf is None:
            return None

        # Create joint configuration dictionary for URDF FK
        urdf_joint_names = [
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        ]

        joint_cfg = {name: joint_array[i] for i, name in enumerate(urdf_joint_names)}

        # Compute foot positions in base frame
        foot_positions_base_frame = self.compute_foot_positions(joint_cfg)
        if foot_positions_base_frame is None or len(foot_positions_base_frame) == 0:
            return None

        # On first call: store initial ground-contact positions
        if not hasattr(self, '_foot_ground_contacts'):
            min_z = min(pos[2] for pos in foot_positions_base_frame.values())
            z_offset = -min_z

            self._foot_ground_contacts = {}
            for foot, pos in foot_positions_base_frame.items():
                adjusted = pos.copy()
                adjusted[2] += z_offset
                self._foot_ground_contacts[foot] = adjusted

            print(f"Storing ground contact positions (lifted by {z_offset:.3f}m to ground plane):")
            for foot, pos in self._foot_ground_contacts.items():
                print(f"  {foot}: {pos}")
            return None  # No adjustment needed on first frame

        # Determine excluded feet from RAISED_FEET dict
        excluded: set[str] = set()
        prim_lower = primitive_name.lower()
        for pattern, feet in self.RAISED_FEET.items():
            if pattern.lower() in prim_lower:
                excluded.update(feet)

        # Build planted-foot arrays
        p_ground_list = []
        p_foot_list = []
        planted_names = []

        for foot_link in self.foot_links:
            if foot_link not in foot_positions_base_frame or foot_link not in self._foot_ground_contacts:
                continue
            if foot_link in excluded:
                continue
            planted_names.append(foot_link)
            p_ground_list.append(self._foot_ground_contacts[foot_link])
            p_foot_list.append(foot_positions_base_frame[foot_link])

        if len(p_ground_list) < 2:
            return None

        p_ground = np.array(p_ground_list)
        p_foot = np.array(p_foot_list)

        # Log contact changes once
        if not hasattr(self, '_prev_planted'):
            self._prev_planted = set()
        current_planted = set(planted_names)
        if current_planted != self._prev_planted:
            raised_names = sorted(excluded & set(self.foot_links))
            print(f"Ground contacts → planted: {planted_names}  raised: {raised_names}")
            self._prev_planted = current_planted

        use_rotation = len(p_ground_list) >= 3

        # Optimisation
        if use_rotation:
            def objective(x):
                tx, ty, tz, roll, pitch, yaw = x
                p_base = np.array([tx, ty, tz])
                R = Rotation.from_euler('xyz', [roll, pitch, yaw]).as_matrix()
                total = 0.0
                for i in range(len(p_ground)):
                    total += np.linalg.norm(p_ground[i] - (p_base + R @ p_foot[i]))**2
                return total

            p_base_init = np.mean(p_ground - p_foot, axis=0)
            x0 = np.concatenate([p_base_init, [0.0, 0.0, 0.0]])
            bounds = [
                (None, None), (None, None), (None, None),
                (-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5),
            ]
        else:
            def objective(x):
                p_base = np.array(x)
                total = 0.0
                for i in range(len(p_ground)):
                    total += np.linalg.norm(p_ground[i] - (p_base + p_foot[i]))**2
                return total

            p_base_init = np.mean(p_ground - p_foot, axis=0)
            x0 = p_base_init
            bounds = [(None, None)] * 3

        options = {'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options=options)

        if not result.success:
            print(f"Warning: Optimization failed at frame {getattr(self, '_debug_counter', 0)}: {result.message}")
            position = p_base_init
            quaternion = np.array([0.0, 0.0, 0.0, 1.0])
        elif use_rotation:
            tx, ty, tz, roll, pitch, yaw = result.x
            position = np.array([tx, ty, tz])
            rot = Rotation.from_euler('xyz', [roll, pitch, yaw])
            quaternion = rot.as_quat()
        else:
            position = np.array(result.x)
            quaternion = np.array([0.0, 0.0, 0.0, 1.0])

        # Apply manual ground offset
        position[2] += self.GROUND_Z_OFFSET

        # Debug output
        if hasattr(self, '_debug_counter'):
            self._debug_counter += 1
        else:
            self._debug_counter = 0
            logging.debug(f"\nFirst optimization result:")
            logging.debug(f"  Position: {position}")
            logging.debug(f"  Quaternion: {quaternion}")
            logging.debug(f"  Planted: {planted_names}")
            if result.success:
                logging.debug(f"  Final error: {result.fun:.6e}")
                logging.debug(f"  Iterations: {result.nit}")

        if self._debug_counter > 0 and self._debug_counter % 100 == 0:
            err_str = f"{result.fun:.6e}" if result.success else 'N/A'
            iter_str = f"{result.nit}" if result.success else 'N/A'
            logging.debug(f"Frame {self._debug_counter}: pos={position}, planted={planted_names}, error={err_str}, iters={iter_str}")

        return (position, quaternion)


def create_robot(robot_name):
    """Factory function to create a Go2 robot instance.

    Args:
        robot_name: Name of the robot (should be ``'go2'``).

    Returns:
        Configured :class:`Go2Robot` with floating-base enabled.
    """
    return Go2Robot(robot_dir=f"robot_{robot_name}", enable_floating_base=True)

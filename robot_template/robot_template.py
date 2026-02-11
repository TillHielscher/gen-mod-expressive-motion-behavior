"""
Template Robot Class

Copy this directory to ``robot_{name}/`` and rename all files accordingly.
Then fill in your robot-specific data following the inline instructions.

Minimal checklist:
  1. Define ``JOINT_NAMES`` and ``DEFAULT_JOINT_ANGLES``.
  2. Place your URDF in ``robot_{name}_description/``.
  3. Record trajectories, convert to DMPs, place in ``robot_{name}_primitives/``.
  4. Fill in ``robot_{name}.yaml`` (capabilities, primitive_lib, parameter_ranges).
  5. Set ``robot: {name}`` in ``config.yaml``.
"""

import sys
import numpy as np
from pathlib import Path

# Add parent directory to path to import RobotBase.
sys.path.insert(0, str(Path(__file__).parent.parent))
from session import RobotBase


class TemplateRobot(RobotBase):
    """Template robot implementation.

    Replace this with a description of your robot, its morphology,
    and any relevant notes about its joints or coordinate conventions.
    """

    # -- Definitions -----------------------------------------------------------

    # Ordered joint names matching the DMP (internal) representation.
    # These must correspond 1-to-1 with columns in your trajectory data.
    JOINT_NAMES = [
        "joint_1",
        "joint_2",
        # ... add all joints
    ]

    # Default pose in radians (same order as JOINT_NAMES).
    DEFAULT_JOINT_ANGLES = np.array([
        0.0,
        0.0,
        # ... one value per joint
    ])

    # Derived lookup table – no need to edit.
    JOINT_NAME_TO_IDX = {name: idx for idx, name in enumerate(JOINT_NAMES)}

    # Number of active joints (derived from JOINT_NAMES).
    N_JOINTS = len(JOINT_NAMES)

    def __init__(self, robot_dir="robot_template"):
        """Initialize the robot.

        Args:
            robot_dir: Path to the robot configuration directory.
        """
        super().__init__(robot_dir)
        self.urdf_path = self.config.get(
            "urdf_path",
            str(self.robot_dir / "robot_template_description" / "robot_template.urdf"),
        )

        self.prepare_handle_rt()

    # -- Robot description related ---------------------------------------------

    def get_joint_names(self):
        """Return ordered list of joint names."""
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
        return {
            name: float(self.DEFAULT_JOINT_ANGLES[i])
            for i, name in enumerate(self.JOINT_NAMES)
        }

    def get_urdf_path(self):
        """Return path to the URDF file."""
        return self.urdf_path

    # -- Real robot ------------------------------------------------------------

    def create_real_session(self):
        pass  # Real robot session creation not implemented.

    def execute_state_on_real_robot(self, state):
        pass  # Real robot execution not implemented.

    # -- Virtual robot ---------------------------------------------------------

    def execute_state_on_virtual_robot(self, virtual_session, state, primitive_name: str = ""):
        """Execute state on the virtual robot.

        Extract your robot's joints from the 48-dim internal DMP state,
        reorder them if necessary, and push to the viewer.

        Args:
            virtual_session: ViserView instance.
            state: Joint state array (48-dimensional internal format).
            primitive_name: Name of the currently playing primitive.
        """
        # Extract your joints from the internal state vector.
        joint_positions = state[: self.N_JOINTS]

        # If your URDF joint order differs from the DMP order, reorder here.
        # Example: converted = self.convert_joint_array_for_virtual(joint_positions)

        virtual_session.set_cfg_array(joint_positions)

    # -- Real time handling ----------------------------------------------------

    def prepare_handle_rt(self):
        """Prepare state needed by :meth:`handle_rt`.

        Set ``self.rt_goal_indices`` to the joint indices whose DMP goals
        are managed in real time (preserved across block transitions).
        Initialise any cycling targets or tracking state here.
        """
        self.rt_goal_indices = []  # No RT-managed joints by default.

    def handle_rt(self, block):
        """Real-time modulation of the current motion block.

        Called at ~20 Hz by the core's RT loop.  Override to implement
        robot-specific behaviour (e.g. head tracking, gaze following).

        Args:
            block: The current ``TimelineBlock``.
        """
        pass

    # -- Helper functions ------------------------------------------------------

    # Add any robot-specific static or instance helpers here.
    # For example, a joint reordering function:
    #
    # @staticmethod
    # def convert_joint_array_for_virtual(input_array):
    #     """Reorder joints from DMP order to URDF order."""
    #     ...


# ---------------------------------------------------------------------------
# Factory function (required by main.py)
# ---------------------------------------------------------------------------

def create_robot(robot_name):
    """Factory function to create a robot instance.

    Args:
        robot_name: Name of the robot (should match the directory suffix).

    Returns:
        Configured robot instance.
    """
    return TemplateRobot(robot_dir=f"robot_{robot_name}")

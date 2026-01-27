# Robot Gen3 Lite - Kinova Gen3 Lite Robot Configuration

This directory contains the robot-specific implementation for the Kinova Gen3 Lite 7-DOF robotic arm.

## Structure

- `robot_gen3_lite.py` - Robot class implementation
- `robot_data.yaml` - Robot capabilities and configuration
- `robot_gen3_lite_primitives/` - Motion primitive library (DMPs)

## Joint Configuration

The Gen3 Lite has 7 revolute joints:
- joint_1: Base rotation
- joint_2: Shoulder
- joint_3: Arm
- joint_4: Forearm 1
- joint_5: Forearm 2
- joint_6: Wrist 1
- joint_7: Wrist 2

Default pose: [0.0, -0.6, 1.9, -1.6, 2.0, 0.0, 0.5]

## Visualization

The Gen3 Lite uses viser for visualization instead of a URDF file. The robot can be visualized using the `robot_descriptions` package with `gen3_lite_description`.

## Converting Trajectories to Primitives

Raw trajectory files from `kinova_viser/*.txt` need to be converted to DMP format:

```python
import numpy as np
import animation_dmp

# Load raw trajectory
demo = np.loadtxt("kinova_viser/wave_arm_120.txt")

# Create DMP
dmp = animation_dmp.DMP(demo, n_basis=100, dt=1/120)

# Save DMP
dmp.save("robot_gen3_lite_primitives/wave_arm")
```

This will create:
- `wave_arm.json` - DMP parameters
- `wave_arm_weights.npy` - Learned weights

## Available Trajectories

In `kinova_viser/`:
- `wave_arm_120.txt` - Waving gesture
- `sway_joyful_120.txt` - Joyful swaying motion
- `point_right.txt` - Pointing gesture
- `indicate_right_120.txt` - Subtle indication gesture
- `default_to_zero.txt` - Return to neutral
- `zero_to_default.txt` - Move to default pose

## Follow-Through Relations

The Gen3 Lite has kinematic follow-through relationships defined in `robot_data.yaml`:
- Wrist joints (5 & 6) follow forearm motion
- Upper arm (joint 3) follows shoulder (joint 2)

These are used for expressive motion modulation.

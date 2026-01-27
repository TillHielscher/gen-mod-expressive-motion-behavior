# Robot Go2 - Unitree Go2 Quadruped Robot Configuration

This directory contains the robot-specific implementation for the Unitree Go2 quadruped robot.

## Structure

- `robot_go2.py` - Robot class implementation
- `robot_data.yaml` - Robot capabilities and configuration
- `robot_go2_primitives/` - Motion primitive library (DMPs)
- `robot_go2_description/` - URDF and mesh files
- `data/` - Raw trajectory files

## Joint Configuration

The Go2 has 12 revolute joints (4 legs × 3 joints per leg):

**DMP Joint Order (indices 0-11):**

**Front Right Leg (FR):**
- FR_hip_joint (FR_0 -> 0)
- FR_thigh_joint (FR_1 -> 1)
- FR_calf_joint (FR_2 -> 2)

**Front Left Leg (FL):**
- FL_hip_joint (FL_0 -> 3)
- FL_thigh_joint (FL_1 -> 4)
- FL_calf_joint (FL_2 -> 5)

**Rear Right Leg (RR):**
- RR_hip_joint (RR_0 -> 6)
- RR_thigh_joint (RR_1 -> 7)
- RR_calf_joint (RR_2 -> 8)

**Rear Left Leg (RL):**
- RL_hip_joint (RL_0 -> 9)
- RL_thigh_joint (RL_1 -> 10)
- RL_calf_joint (RL_2 -> 11)

Default standing pose (radians):
```
[0.0, 0.78, -1.41,   # FR
 0.0, 0.78, -1.40,   # FL
 0.09, 0.78, -1.39,  # RR
 0.0, 0.81, -1.45]   # RL
```

## Visualization

The Go2 uses its URDF file located at `robot_go2_description/go2_description.urdf` for visualization.

### Floating-Base Visualization

**Problem:** Standard URDF visualization fixes the robot's base (torso) in space and moves the legs. For a quadruped, this is physically incorrect - the feet should stay planted on the ground while the torso moves.

**Solution:** The Go2Robot class implements floating-base visualization:
1. Computes forward kinematics to determine foot positions
2. Tracks reference foot positions from the initial pose
3. Calculates the base displacement needed to keep feet planted
4. Updates the visualization base frame to compensate

This creates realistic motion where the feet appear planted and the torso moves naturally, matching real quadruped locomotion.

**Note:** This feature is Go2-specific and doesn't affect fixed-base robots (Pepper, Gen3 Lite) which remain backward compatible.

## Converting Trajectories to Primitives

Raw trajectory files from `data/*.txt` need to be converted to DMP format:

```bash
cd robot_go2
python convert_trajectories.py
```

This will create DMP files:
- `point_right.json` + `point_right_weights.npy`
- `tilt_left.json` + `tilt_left_weights.npy`

## Available Trajectories

In `data/`:
- `point_right.txt` - Body pointing/leaning to the right
- `tilt_left.txt` - Body tilting to the left

## Follow-Through Relations

The Go2 has leg kinematic follow-through relationships defined in `robot_data.yaml`:
- Each leg's calf joint follows its thigh joint (natural leg movement)
- This creates more fluid and natural quadruped motion

## Usage

To use the Go2 robot:

1. Convert trajectories (if not already done):
   ```bash
   cd robot_go2
   python convert_trajectories.py
   ```

2. Set robot in `config.yaml`:
   ```yaml
   robot: go2
   ```

3. Run the system:
   ```bash
   python main.py
   ```

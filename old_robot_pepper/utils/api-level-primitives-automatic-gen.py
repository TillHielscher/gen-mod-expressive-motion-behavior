import json
import numpy as np
from pathlib import Path

# Idle joint positions
stand_init_joint_angles = {
    "HeadPitch": -10.0,
    "HipPitch": -2.0,
    "HipRoll": 0.0,
    "KneePitch": 0.0,
    "LElbowRoll": -30.0,
    "LElbowYaw": -70.0,
    "LShoulderPitch": 90.0,
    "LShoulderRoll": 10.0,
    "LWristYaw": 0.0,
    "RElbowRoll": 30.0,
    "RElbowYaw": 70.0,
    "RShoulderPitch": 90.0,
    "RShoulderRoll": -10.0,
    "RWristYaw": 0.0,
    "HeadYaw": 0.0,
}

# Primitive → (joint, target angle)
primitives = {
    "raise_left_arm_forward_180": ("LShoulderPitch", -80.0),
    "raise_left_arm_forward_90": ("LShoulderPitch", 0.0),
    "raise_left_arm_forward_45": ("LShoulderPitch", 45.0),
    "raise_right_arm_forward_180": ("RShoulderPitch", -80.0),
    "raise_right_arm_forward_90": ("RShoulderPitch", 0.0),
    "raise_right_arm_forward_45": ("RShoulderPitch", 45.0),
    "raise_left_arm_side_90": ("LShoulderRoll", 89.5),
    "raise_left_arm_side_45": ("LShoulderRoll", 45.0),
    "raise_right_arm_side_90": ("RShoulderRoll", -89.5),
    "raise_right_arm_side_45": ("RShoulderRoll", -45.0),
    "turn_head_left_90": ("HeadYaw", 90.0),
    "turn_head_left_45": ("HeadYaw", 45.0),
    "turn_head_right_90": ("HeadYaw", -90.0),
    "turn_head_right_45": ("HeadYaw", -45.0),
    "tilt_head_up": ("HeadPitch", -40.5),
    "tilt_head_down": ("HeadPitch", 25.5),
    "lean_left_30": ("HipRoll", 29.5),
    "lean_left_15": ("HipRoll", 15.0),
    "lean_right_30": ("HipRoll", -29.5),
    "lean_right_15": ("HipRoll", -15.0),
    "lean_forward_45": ("HipPitch", 45.0),
    "lean_forward_20": ("HipPitch", 20.0),
    "lean_backward_45": ("HipPitch", -45.0),
    "lean_backward_20": ("HipPitch", -20.0),
    "angle_left_arm_90": ("LElbowRoll", -89.5),
    "angle_left_arm_45": ("LElbowRoll", -45.0),
    "angle_right_arm_90": ("RElbowRoll", 89.5),
    "angle_right_arm_45": ("RElbowRoll", 45.0),
}

# Output directory
out_dir = Path("robot_primitives_json")
out_dir.mkdir(parents=True, exist_ok=True)

def generate_range(start, end, step=1.0):
    """Generate linear list of angles with 1° increments"""
    if start < end:
        values = np.arange(start, end + step, step)
    else:
        values = np.arange(start, end - step, -step)
    return [round(float(v), 2) for v in values]

def generate_round_trip_range(start, end, step=1.0):
    """Generate linear list of angles that goes to target and back"""
    # Forward trajectory
    if start < end:
        forward = np.arange(start, end + step, step)
    else:
        forward = np.arange(start, end - step, -step)
    
    # Return trajectory (excluding the end point to avoid duplicate)
    if end < start:
        backward = np.arange(end + step, start + step, step)
    else:
        backward = np.arange(end - step, start - step, -step)
    
    # Combine forward and backward trajectories
    values = np.concatenate([forward, backward])
    return [round(float(v), 2) for v in values]

# Create one JSON file per primitive
for prim, (joint, target) in primitives.items():
    start = stand_init_joint_angles[joint]
    #values = generate_range(start, target, step=1.0)
    values = generate_round_trip_range(start, target, step=0.75)
    data = {joint: values}
    
    filepath = out_dir / f"{prim}.json"
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)

print(f"✅ JSON files saved to: {out_dir.resolve()}")

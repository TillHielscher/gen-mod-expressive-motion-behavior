"""
Script to convert old pickle files (DMPs) to JSON format.

This script uses the old animation_dmp package to load pickle files
and converts them to JSON format with joint trajectories.

JSON format:
{
    "Joint1": [val_t0, val_t1, ...],
    "Joint2": [val_t0, val_t1, ...],
    ...
}
"""

import os
import sys
import pickle
import json
import numpy as np
from pathlib import Path

# Joint names in the order they appear in the DMP state (48 dimensions)
joint_names = [
    "KneePitch",
    "HipPitch",
    "HipRoll",
    "HeadYaw",
    "HeadPitch",
    "LShoulderPitch",
    "LShoulderRoll",
    "LElbowYaw",
    "LElbowRoll",
    "LWristYaw",
    "LHand",
    "LFinger21",
    "LFinger22",
    "LFinger23",
    "LFinger11",
    "LFinger12",
    "LFinger13",
    "LFinger41",
    "LFinger42",
    "LFinger43",
    "LFinger31",
    "LFinger32",
    "LFinger33",
    "LThumb1",
    "LThumb2",
    "RShoulderPitch",
    "RShoulderRoll",
    "RElbowYaw",
    "RElbowRoll",
    "RWristYaw",
    "RHand",
    "RFinger41",
    "RFinger42",
    "RFinger43",
    "RFinger31",
    "RFinger32",
    "RFinger33",
    "RFinger21",
    "RFinger22",
    "RFinger23",
    "RFinger11",
    "RFinger12",
    "RFinger13",
    "RThumb1",
    "RThumb2",
    "WheelFL",
    "WheelB",
    "WheelFR"
]


def replay_dmp_to_trajectory(dmp, num_steps=None, hz=60):
    """
    Replay a DMP and collect the trajectory.
    
    Args:
        dmp: The DMP object loaded from pickle
        num_steps: Number of steps to replay (if None, use tau * hz)
        hz: Frequency in Hz
        
    Returns:
        numpy array of shape (num_steps, 48) with joint values over time
    """
    # Reset DMP to initial state
    dmp.init_state()

    trajectory = dmp.run()
    
    """ # Determine number of steps
    if num_steps is None:
        tau = dmp.get_state()["tau"]
        num_steps = int(tau * hz)
    
    # Collect trajectory
    trajectory = []
    dt = 1.0 / hz
    
    for i in range(num_steps):
        # Step the DMP
        dmp.step(dt)
        
        # Get current state
        state = dmp.get_state()
        y = state["y"]  # This should be the 48-dimensional joint state
        
        trajectory.append(y.copy())
        
        # Check if DMP is complete
        if state["t"] >= state["tau"]:
            break """
    
    return np.array(trajectory)


def trajectory_to_json_dict(trajectory):
    """
    Convert trajectory array to JSON-compatible dictionary.
    
    Args:
        trajectory: numpy array of shape (num_steps, 48)
        
    Returns:
        dict with joint names as keys and lists of values
    """
    if trajectory.shape[1] != 48:
        raise ValueError(f"Expected 48 joints, got {trajectory.shape[1]}")
    
    json_dict = {}
    
    for i, joint_name in enumerate(joint_names):
        # Convert to degrees and create list
        values = np.rad2deg(trajectory[:, i]).tolist()
        json_dict[joint_name] = values
    
    return json_dict


def convert_pickle_to_json(pickle_path, json_path, hz=60, num_steps=None):
    """
    Convert a single pickle file to JSON.
    
    Args:
        pickle_path: Path to input pickle file
        json_path: Path to output JSON file
        hz: Replay frequency in Hz
        num_steps: Number of steps (if None, calculated from tau)
    """
    print(f"Converting: {pickle_path}")
    
    try:
        # Load pickle file
        with open(pickle_path, 'rb') as f:
            dmp = pickle.load(f)
        
        # Replay DMP to get trajectory
        trajectory = replay_dmp_to_trajectory(dmp, num_steps=num_steps, hz=hz)
        
        print(f"  Generated trajectory: {trajectory.shape[0]} timesteps, {trajectory.shape[1]} joints")
        
        # Convert to JSON format
        json_dict = trajectory_to_json_dict(trajectory)
        
        # Save to JSON
        with open(json_path, 'w') as f:
            json.dump(json_dict, f, indent=2)
        
        print(f"  Saved to: {json_path}")
        return True
        
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def convert_folder(input_folder, output_folder, hz=60, num_steps=None):
    """
    Convert all pickle files in a folder to JSON.
    
    Args:
        input_folder: Folder containing pickle files
        output_folder: Folder to save JSON files
        hz: Replay frequency in Hz
        num_steps: Number of steps (if None, calculated from tau)
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    
    # Create output folder if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all pickle files
    pickle_files = sorted(input_path.glob("*.pickle"))
    
    if not pickle_files:
        print(f"No pickle files found in {input_folder}")
        return
    
    print(f"Found {len(pickle_files)} pickle files")
    print(f"Output folder: {output_folder}")
    print("=" * 60)
    
    success_count = 0
    fail_count = 0
    
    for pickle_file in pickle_files:
        # Create corresponding JSON filename
        json_filename = pickle_file.stem + ".json"
        json_path = output_path / json_filename
        
        # Convert
        if convert_pickle_to_json(str(pickle_file), str(json_path), hz=hz, num_steps=num_steps):
            success_count += 1
        else:
            fail_count += 1
        
        print()
    
    print("=" * 60)
    print(f"Conversion complete!")
    print(f"  Success: {success_count}")
    print(f"  Failed:  {fail_count}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Convert pickle DMP files to JSON trajectories',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert all primitives in a folder
  python replay_primitive_to_json.py \\
      --input robot_pepper/primitive_admp_picked \\
      --output robot_pepper/primitive_json
  
  # Convert API level primitives
  python replay_primitive_to_json.py \\
      --input robot_pepper/api_level_primitive_admp_picked \\
      --output robot_pepper/api_level_primitive_json
  
  # Convert a single file
  python replay_primitive_to_json.py \\
      --input robot_pepper/primitive_admp_picked/wave_arm.pickle \\
      --output wave_arm.json
  
  # Specify replay frequency and number of steps
  python replay_primitive_to_json.py \\
      --input robot_pepper/primitive_admp_picked \\
      --output robot_pepper/primitive_json \\
      --hz 30 \\
      --steps 100
        """
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input pickle file or folder containing pickle files'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output JSON file or folder for JSON files'
    )
    
    parser.add_argument(
        '--hz',
        type=int,
        default=60,
        help='Replay frequency in Hz (default: 60)'
    )
    
    parser.add_argument(
        '--steps',
        type=int,
        default=None,
        help='Number of steps to replay (default: calculated from tau)'
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    # Check if input is a file or folder
    if input_path.is_file():
        # Single file conversion
        if not input_path.suffix == '.pickle':
            print(f"Error: Input file must be a .pickle file")
            sys.exit(1)
        
        output_path = Path(args.output)
        if output_path.is_dir():
            # If output is a directory, create filename based on input
            output_path = output_path / (input_path.stem + ".json")
        
        convert_pickle_to_json(str(input_path), str(output_path), hz=args.hz, num_steps=args.steps)
        
    elif input_path.is_dir():
        # Folder conversion
        convert_folder(str(input_path), args.output, hz=args.hz, num_steps=args.steps)
        
    else:
        print(f"Error: Input path does not exist: {args.input}")
        sys.exit(1)


if __name__ == "__main__":
    main()

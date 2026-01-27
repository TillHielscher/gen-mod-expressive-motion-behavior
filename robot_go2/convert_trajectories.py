#!/usr/bin/env python3
"""
Convert raw trajectory files from data/ to DMP primitives.

Usage:
    python convert_trajectories.py
"""

import numpy as np
import animation_dmp
from pathlib import Path

# Define trajectory files to convert
trajectories = {
    "point_right": "data/point_right.txt",
    "tilt_left": "data/tilt_left.txt",
}

def convert_trajectory_to_dmp(input_file, output_name, n_basis=100, hz=60):
    """
    Convert a raw trajectory file to DMP format.
    
    Args:
        input_file: Path to input .txt file
        output_name: Name for output DMP (without extension)
        n_basis: Number of basis functions
        hz: Sampling frequency (estimated from data)
    """
    print(f"Converting {input_file} -> {output_name}")
    
    # Load trajectory - Go2 has 12 joints per timestep (4 legs × 3 joints)
    data = np.loadtxt(input_file)
    
    # The data appears to be in format: [timesteps, 12 values per timestep]
    # Reshape if needed
    if len(data.shape) == 1:
        # Single row, reshape
        num_joints = 12
        data = data.reshape(-1, num_joints)
    elif data.shape[1] != 12:
        # May need to extract 12 values per row
        print(f"  Warning: Expected 12 joints, got {data.shape[1]} columns")
        if data.shape[1] % 12 == 0:
            # Might be multiple poses per row
            data = data.reshape(-1, 12)
    
    print(f"  Trajectory shape: {data.shape}")
    
    # Create DMP
    dmp = animation_dmp.DMP(data, n_weights_dim=n_basis, dt=1/hz)
    
    # Save DMP
    output_path = Path("robot_go2_primitives") / output_name
    dmp.save(str(output_path))
    print(f"  Saved to {output_path}.json and {output_path}_weights.npy")


def main():
    """Convert all trajectories."""
    print("Converting Go2 trajectories to DMP format...")
    print("=" * 60)
    
    for name, input_file in trajectories.items():
        input_path = Path(input_file)
        if input_path.exists():
            try:
                convert_trajectory_to_dmp(input_file, name)
                print()
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
                print()
        else:
            print(f"SKIPPING {name}: File not found at {input_file}")
            print()
    
    print("=" * 60)
    print("Conversion complete!")


if __name__ == "__main__":
    main()

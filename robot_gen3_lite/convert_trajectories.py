#!/usr/bin/env python3
"""
Convert raw trajectory files from kinova_viser to DMP primitives.

Usage:
    python convert_trajectories.py
"""

import numpy as np
import animation_dmp
from pathlib import Path

# Define trajectory files to convert
# trajectories = {
#     "wave_arm": "../kinova_viser/wave_arm_120.txt",
#     "sway_joyful": "../kinova_viser/sway_joyful_120.txt",
#     "point_right": "../kinova_viser/point_right.txt",
#     "indicate_right": "../kinova_viser/indicate_right_120.txt",
#     "default_to_zero": "../kinova_viser/default_to_zero.txt",
#     "zero_to_default": "../kinova_viser/zero_to_default.txt",
# }
trajectories = {
    "default": "../kinova_viser/default.txt",
}

def convert_trajectory_to_dmp(input_file, output_name, n_basis=100, hz=120):
    """
    Convert a raw trajectory file to DMP format.
    
    Args:
        input_file: Path to input .txt file
        output_name: Name for output DMP (without extension)
        n_basis: Number of basis functions
        hz: Sampling frequency
    """
    print(f"Converting {input_file} -> {output_name}")
    
    # Load trajectory
    demo = np.loadtxt(input_file)
    print(f"  Trajectory shape: {demo.shape}")
    
    # Create DMP
    dmp = animation_dmp.DMP(demo, n_weights_dim=n_basis, dt=1/hz)
    
    # Save DMP
    output_path = Path("robot_gen3_lite_primitives") / output_name
    dmp.save(str(output_path))
    print(f"  Saved to {output_path}.json and {output_path}_weights.npy")


def main():
    """Convert all trajectories."""
    print("Converting Gen3 Lite trajectories to DMP format...")
    print("=" * 60)
    
    for name, input_file in trajectories.items():
        input_path = Path(input_file)
        if input_path.exists():
            try:
                convert_trajectory_to_dmp(input_file, name)
                print()
            except Exception as e:
                print(f"  ERROR: {e}")
                print()
        else:
            print(f"SKIPPING {name}: File not found at {input_file}")
            print()
    
    print("=" * 60)
    print("Conversion complete!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Convert raw trajectory files from data/ to DMP primitives.

Usage:
    python convert_trajectories.py
"""

import numpy as np
import animation_dmp
from pathlib import Path


def go2_trajectory_to_demo(trajectory_path: Path, sample_dt=1 / 60):
    """
    Load and resample Go2 trajectory data.
    
    Args:
        trajectory_path: Path to directory containing q.txt and tick.txt
        sample_dt: Target sample time in seconds (default: 1/60)
        
    Returns:
        Dictionary with resampled q and t arrays
    """
    q = np.loadtxt(trajectory_path / "q.txt", dtype=np.float32)
    ticks = np.loadtxt(trajectory_path / "tick.txt", dtype=np.float32)

    print(f"Total number of samples {len(ticks)}")

    # ticks is ms
    ticks_sec = ticks / 1000.0

    t_min = ticks_sec[0]
    t_max = ticks_sec[-1]
    desired_times = np.arange(t_min, t_max, sample_dt)

    indices = np.searchsorted(ticks_sec, desired_times)
    indices = np.clip(indices, 0, len(ticks_sec) - 1)

    print(f"Subsampled to {len(indices)} samples")

    return {"q": q[indices], "t": desired_times}


# Define trajectory directories to convert (now using q.txt + tick.txt format)
trajectory_dirs = {
    "dance": "data/dance",
    "default": "data/default",
    "look_right": "data/look_right",
    "point_right": "data/point_right",
    "shake_hand": "data/shake_hand",
    "tilt_left": "data/tilt_left",
}


def convert_demo_dict_to_dmp(demo_dict, output_name, n_basis=100):
    """
    Convert a demo dictionary (from go2_trajectory_to_demo) to DMP format.
    
    Args:
        demo_dict: Dictionary with keys 'q', 't' from go2_trajectory_to_demo
        output_name: Name for output DMP (without extension)
        n_basis: Number of basis functions
    """
    print(f"Converting demo dict -> {output_name}")
    
    # Extract ONLY the joint positions (q)
    q = demo_dict["q"]
    t = demo_dict["t"]
    
    print(f"  Joint positions shape: {q.shape}")
    print(f"  Number of timesteps: {len(t)}")
    
    # Calculate dt from time array
    dt = np.mean(np.diff(t))
    print(f"  Average dt: {dt:.6f} seconds ({1/dt:.1f} Hz)")
    
    # Verify we have 12 joints
    if q.shape[1] != 12:
        raise ValueError(f"Expected 12 joints, got {q.shape[1]}")
    
    # Create DMP using ONLY joint positions
    dmp = animation_dmp.DMP(q, n_weights_dim=n_basis, dt=dt)
    
    # Save DMP
    output_path = Path("robot_go2_primitives") / output_name
    dmp.save(str(output_path))
    print(f"  Saved to {output_path}.json and {output_path}_weights.npy")


def main():
    """Convert all trajectories using the demo format."""
    print("Converting Go2 trajectories to DMP format...")
    print("=" * 60)
    
    for name, traj_dir in trajectory_dirs.items():
        traj_path = Path(traj_dir)
        
        # Check if directory exists and has required files
        q_file = traj_path / "q.txt"
        tick_file = traj_path / "tick.txt"
        
        if not traj_path.exists():
            print(f"SKIPPING {name}: Directory not found at {traj_path}")
            print()
            continue
            
        if not q_file.exists() or not tick_file.exists():
            print(f"SKIPPING {name}: Missing q.txt or tick.txt in {traj_path}")
            print()
            continue
        
        try:
            # Load and resample trajectory
            demo = go2_trajectory_to_demo(traj_path, sample_dt=1/60)
            
            # Convert to DMP
            convert_demo_dict_to_dmp(demo, name, n_basis=100)
            print()
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            print()
    
    print("=" * 60)
    print("Conversion complete!")


if __name__ == "__main__":
    main()

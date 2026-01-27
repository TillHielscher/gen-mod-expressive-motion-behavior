"""
Test script for Go2 floating-base visualization.

This script tests that:
1. Go2 robot loads correctly
2. FK computation works
3. Base transform is computed properly
4. Virtual visualization shows feet planted (not torso fixed)
"""

import numpy as np
import time
from robot_go2.robot_go2 import Go2Robot
from llag_virtual_session import VirtualSession


def test_floating_base():
    """Test the floating-base visualization with Go2."""
    print("=" * 60)
    print("Go2 Floating-Base Visualization Test")
    print("=" * 60)
    
    # Create Go2 robot
    print("\n1. Creating Go2 robot instance...")
    go2 = Go2Robot()
    print(f"   URDF path: {go2.get_urdf_path()}")
    print(f"   Joint names: {go2.get_joint_names()}")
    
    # Create virtual session
    print("\n2. Starting virtual session...")
    urdf_path = go2.get_urdf_path()
    
    # Get default joint angles in URDF order (need to convert from DMP order)
    default_dmp = go2.DEFAULT_JOINT_ANGLES
    default_urdf = go2.convert_joint_array_for_virtual(default_dmp)
    
    # Create joint dict for initial pose
    joint_names_urdf = [
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    ]
    initial_joints = {name: default_urdf[i] for i, name in enumerate(joint_names_urdf)}
    
    virtual = VirtualSession(
        urdf_path=urdf_path,
        initial_joint_angles=initial_joints,
        load_meshes=True,
        port=8088
    )
    print("   Virtual session started on port 8088")
    print("   Open http://localhost:8088 in your browser")
    
    # Wait for client
    print("\n3. Waiting for client connection...")
    if not virtual.wait_for_client(timeout=30):
        print("   No client connected. Exiting.")
        return
    
    print("   Client connected!")
    
    # Test sequence: animate through different poses
    print("\n4. Testing floating-base behavior...")
    print("   Watch how the TORSO moves while feet stay planted!")
    print("   (Without floating-base, the torso would stay fixed)")
    
    # Create internal state (48-dim) with default pose
    base_state = np.zeros(48)
    base_state[:12] = default_dmp
    
    # Test poses - vary thigh and calf angles to shift weight
    test_poses = [
        # Start with default
        ("Default standing pose", default_dmp.copy()),
        
        # Shift weight forward (increase front thigh angles)
        ("Weight forward", default_dmp.copy()),
        
        # Shift weight back (increase rear thigh angles)
        ("Weight backward", default_dmp.copy()),
        
        # Crouch (increase all thigh angles)
        ("Crouch down", default_dmp.copy()),
        
        # Back to default
        ("Return to default", default_dmp.copy()),
    ]
    
    # Modify test poses
    test_poses[1][1][[1, 4]] += 0.3  # Front legs: increase thigh
    test_poses[2][1][[7, 10]] += 0.3  # Rear legs: increase thigh
    test_poses[3][1][[1, 4, 7, 10]] += 0.3  # All legs: crouch
    
    # Animate through poses
    for pose_name, joint_config in test_poses:
        print(f"\n   -> {pose_name}")
        state = np.zeros(48)
        state[:12] = joint_config
        
        # Execute on virtual robot (this will compute base transform)
        go2.execute_state_on_virtual_robot(virtual, state)
        
        time.sleep(2.0)
    
    print("\n5. Test complete!")
    print("   The robot should have moved its torso while keeping feet planted.")
    print("   Press Ctrl+C to exit.")
    
    # Keep running
    try:
        virtual.run_forever()
    except KeyboardInterrupt:
        print("\n\nExiting...")


if __name__ == "__main__":
    test_floating_base()

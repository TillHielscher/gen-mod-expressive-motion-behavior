"""
Main entry point for the LLAG (Language-Guided Animation Generation) system.

This version uses the robot-agnostic llag_virtual_session module.
"""

import asyncio
import argparse
from llag_core import LLAGCore


def create_pepper_core(args):
    """Create LLAGCore configured for Pepper robot."""
    return LLAGCore(
        use_pepper=args.use_real_robot,
        use_virtual_pepper=args.use_virtual,
        console_log=True,
        debug_log=args.debug,
        short_pipeline=args.short_pipeline,
        modulate=args.modulate,
        primitive_lib_path="robot_pepper/primitive_saved",
        robot_description_path="robot_pepper/robot_data.yaml",
        prompt_data_path="prompts_v4.yaml",
        urdf_path="pepper-toolbox-main/src/peppertoolbox/urdf/pepper_pruned.urdf"
    )


def create_go2_core(args):
    """Create LLAGCore configured for Go2 quadruped robot."""
    return LLAGCore(
        use_pepper=False,  # Go2 doesn't use pepper hardware
        use_virtual_pepper=args.use_virtual,
        console_log=True,
        debug_log=args.debug,
        short_pipeline=args.short_pipeline,
        modulate=args.modulate,
        primitive_lib_path="robot_go2/primitive_admp_picked",
        robot_description_path="robot_go2/robot_data.yaml",
        prompt_data_path="prompts_v4.yaml",
        urdf_path="kinematic_analysis/go2_urdf/go2_description.urdf"
    )


def create_franka_core(args):
    """Create LLAGCore configured for Franka robot."""
    return LLAGCore(
        use_pepper=False,  # Franka doesn't use pepper hardware
        use_virtual_pepper=args.use_virtual,
        console_log=True,
        debug_log=args.debug,
        short_pipeline=args.short_pipeline,
        modulate=args.modulate,
        primitive_lib_path="robot_franka/primitive_admp_picked",
        robot_description_path="robot_franka/robot_data.yaml",
        prompt_data_path="prompts_v4.yaml",
        urdf_path="kinematic_analysis/franka_urdf/panda.urdf"
    )


def create_custom_core(args):
    """Create LLAGCore with custom configuration from command line."""
    return LLAGCore(
        use_pepper=args.use_real_robot,
        use_virtual_pepper=args.use_virtual,
        console_log=True,
        debug_log=args.debug,
        short_pipeline=args.short_pipeline,
        modulate=args.modulate,
        primitive_lib_path=args.primitive_lib_path,
        robot_description_path=args.robot_description_path,
        prompt_data_path=args.prompt_data_path,
        urdf_path=args.urdf_path
    )


def main():
    parser = argparse.ArgumentParser(
        description='LLAG - Language-Guided Animation Generation System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with Pepper robot (virtual visualization)
  python main.py --robot pepper
  
  # Run with Go2 quadruped robot
  python main.py --robot go2
  
  # Run with Franka robot
  python main.py --robot franka
  
  # Run with custom robot configuration
  python main.py --robot custom \\
      --urdf-path path/to/robot.urdf \\
      --primitive-lib robot_name/primitive_admp_picked \\
      --robot-description robot_name/robot_data.yaml
  
  # Run with real Pepper hardware (requires connection)
  python main.py --robot pepper --use-real-robot
  
  # Run in debug mode
  python main.py --robot pepper --debug
  
        """
    )
    
    # Robot selection
    parser.add_argument(
        '--robot',
        type=str,
        default='pepper',
        choices=['pepper', 'go2', 'franka', 'custom'],
        help='Select which robot to use (default: pepper)'
    )
    
    # Basic operation modes
    parser.add_argument(
        '--use-real-robot',
        action='store_true',
        help='Use real robot hardware (currently only supports Pepper)'
    )
    
    parser.add_argument(
        '--use-virtual',
        action='store_true',
        default=True,
        help='Use virtual robot visualization (default: True)'
    )
    
    parser.add_argument(
        '--no-virtual',
        action='store_false',
        dest='use_virtual',
        help='Disable virtual robot visualization'
    )
    
    # Pipeline configuration
    parser.add_argument(
        '--short-pipeline',
        action='store_true',
        default=True,
        help='Use short planning pipeline (default: True)'
    )
    
    parser.add_argument(
        '--long-pipeline',
        action='store_false',
        dest='short_pipeline',
        help='Use long planning pipeline'
    )
    
    parser.add_argument(
        '--no-modulate',
        action='store_false',
        dest='modulate',
        default=True,
        help='Disable animation modulation'
    )
    
    
    # Debugging
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    # Custom robot configuration (used when --robot custom)
    parser.add_argument(
        '--urdf-path',
        type=str,
        default='pepper-toolbox-main/src/peppertoolbox/urdf/pepper_pruned.urdf',
        help='Path to robot URDF file (for custom robot)'
    )
    
    parser.add_argument(
        '--primitive-lib-path',
        type=str,
        default='robot_pepper/primitive_saved',
        dest='primitive_lib_path',
        help='Path to primitive library (for custom robot)'
    )
    
    parser.add_argument(
        '--robot-description-path',
        type=str,
        default='robot_pepper/robot_data.yaml',
        dest='robot_description_path',
        help='Path to robot description YAML (for custom robot)'
    )
    
    parser.add_argument(
        '--prompt-data-path',
        type=str,
        default='prompts_v4.yaml',
        dest='prompt_data_path',
        help='Path to prompt data YAML'
    )
    
    args = parser.parse_args()
    
    # Create appropriate core based on robot selection
    print(f"=== Initializing LLAG System ===")
    print(f"Robot: {args.robot}")
    print(f"Virtual visualization: {args.use_virtual}")
    print(f"Real robot: {args.use_real_robot}")
    print(f"Pipeline: {'short' if args.short_pipeline else 'long'}")
    print(f"Modulation: {args.modulate}")
    print(f"Debug mode: {args.debug}")
    print()
    
    if args.robot == 'pepper':
        core = create_pepper_core(args)
    elif args.robot == 'go2':
        core = create_go2_core(args)
    elif args.robot == 'franka':
        core = create_franka_core(args)
    elif args.robot == 'custom':
        print(f"Custom configuration:")
        print(f"  URDF: {args.urdf_path}")
        print(f"  Primitives: {args.primitive_lib_path}")
        print(f"  Description: {args.robot_description_path}")
        print()
        core = create_custom_core(args)
    else:
        raise ValueError(f"Unknown robot type: {args.robot}")
    
    # Run the system
    print("Starting LLAG system...")
    print("=" * 50)
    try:
        asyncio.run(core.run())
    except KeyboardInterrupt:
        print("\n\nShutting down gracefully...")
    except Exception as e:
        print(f"\n\nError occurred: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

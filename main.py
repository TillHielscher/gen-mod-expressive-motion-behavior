"""
Main entry point for the LLAG (Language-Guided Animation Generation) system.

This version uses the robot-agnostic architecture with robot classes.
"""

import asyncio
import yaml
from pathlib import Path
from llag_core import LLAGCore
from robot_pepper.robot_pepper import create_robot


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def create_pepper_core(config: dict):
    """Create LLAGCore configured for Pepper robot."""
    robot = create_robot('pepper')
    return LLAGCore(
        robot=robot,
        use_real_robot=config['use_real_robot'],
        use_virtual_robot=config['use_virtual_robot'],
        console_log=True,
        debug_log=config['debug'],
        short_pipeline=config['short_pipeline'],
        modulate=config['modulate'],
        prompt_data_path=config['prompt_data_path']
    )


def create_go2_core(config: dict):
    """Create LLAGCore configured for Go2 quadruped robot."""
    # TODO: Implement Go2Robot class
    # robot = create_robot('go2')
    # For now, use the old approach
    from robot_pepper.robot_pepper import PepperRobot
    
    # Create a temporary robot class wrapper for Go2
    class TempGo2Robot(PepperRobot):
        def __init__(self):
            # Don't call super().__init__() with default path
            from pathlib import Path
            self.robot_dir = Path(config['robots']['go2']['robot_dir'])
            self.config = self._load_config()
            self.urdf_path = config['robots']['go2']['urdf_path']
    
    robot = TempGo2Robot()
    return LLAGCore(
        robot=robot,
        use_real_robot=False,
        use_virtual_robot=config['use_virtual_robot'],
        console_log=True,
        debug_log=config['debug'],
        short_pipeline=config['short_pipeline'],
        modulate=config['modulate'],
        prompt_data_path=config['prompt_data_path']
    )


def create_franka_core(config: dict):
    """Create LLAGCore configured for Franka robot."""
    # TODO: Implement FrankaRobot class
    # robot = create_robot('franka')
    # For now, use the old approach
    from robot_pepper.robot_pepper import PepperRobot
    from pathlib import Path
    
    class TempFrankaRobot(PepperRobot):
        def __init__(self):
            self.robot_dir = Path(config['robots']['franka']['robot_dir'])
            self.config = self._load_config()
            self.urdf_path = config['robots']['franka']['urdf_path']
    
    robot = TempFrankaRobot()
    return LLAGCore(
        robot=robot,
        use_real_robot=False,
        use_virtual_robot=config['use_virtual_robot'],
        console_log=True,
        debug_log=config['debug'],
        short_pipeline=config['short_pipeline'],
        modulate=config['modulate'],
        prompt_data_path=config['prompt_data_path']
    )


def create_custom_core(config: dict):
    """Create LLAGCore with custom configuration from config file."""
    from robot_pepper.robot_pepper import PepperRobot
    from pathlib import Path
    import os
    
    custom_config = config['custom_robot']
    robot_dir = os.path.dirname(custom_config['robot_description_path'])
    
    class CustomRobot(PepperRobot):
        def __init__(self, robot_dir_path, urdf):
            self.robot_dir = Path(robot_dir_path)
            self.config = self._load_config()
            self.urdf_path = urdf
    
    robot = CustomRobot(robot_dir, custom_config['urdf_path'])
    return LLAGCore(
        robot=robot,
        use_real_robot=config['use_real_robot'],
        use_virtual_robot=config['use_virtual_robot'],
        console_log=True,
        debug_log=config['debug'],
        short_pipeline=config['short_pipeline'],
        modulate=config['modulate'],
        prompt_data_path=config['prompt_data_path']
    )


def main():
    """Main entry point for the LLAG system."""
    # Load configuration
    config = load_config("config.yaml")
    
    robot_type = config['robot']
    
    # Print configuration summary
    print(f"=== Initializing LLAG System ===")
    print(f"Robot: {robot_type}")
    print(f"Virtual visualization: {config['use_virtual_robot']}")
    print(f"Real robot: {config['use_real_robot']}")
    print(f"Pipeline: {'short' if config['short_pipeline'] else 'long'}")
    print(f"Modulation: {config['modulate']}")
    print(f"Debug mode: {config['debug']}")
    print()
    
    # Create appropriate core based on robot selection
    if robot_type == 'pepper':
        core = create_pepper_core(config)
    elif robot_type == 'go2':
        core = create_go2_core(config)
    elif robot_type == 'franka':
        core = create_franka_core(config)
    elif robot_type == 'custom':
        custom = config['custom_robot']
        print(f"Custom configuration:")
        print(f"  URDF: {custom['urdf_path']}")
        print(f"  Primitives: {custom['primitive_lib_path']}")
        print(f"  Description: {custom['robot_description_path']}")
        print()
        core = create_custom_core(config)
    else:
        raise ValueError(f"Unknown robot type: {robot_type}")
    
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

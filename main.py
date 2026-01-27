"""
Main entry point for the LLAG (Language-Guided Animation Generation) system.

This version uses the robot-agnostic architecture with robot classes.
"""

import asyncio
import yaml
import importlib
from pathlib import Path
from llag_core import LLAGCore


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def create_robot_instance(config: dict):
    """
    Create robot instance based on configuration.
    
    Dynamically imports robot_{name}.robot_{name} module and calls its create_robot() function.
    Expected structure: robot_{name}/robot_{name}.py with create_robot() function.
    """
    robot_name = config['robot']
    
    try:
        # Dynamically import the robot module
        module_name = f"robot_{robot_name}.robot_{robot_name}"
        robot_module = importlib.import_module(module_name)
        
        # Get the create_robot function from the module
        if not hasattr(robot_module, 'create_robot'):
            raise AttributeError(
                f"Module '{module_name}' does not have a 'create_robot' function. "
                f"Please ensure robot_{robot_name}/robot_{robot_name}.py defines create_robot()."
            )
        
        create_robot = robot_module.create_robot
        
        # Call create_robot with the robot name
        return create_robot(robot_name)
        
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            f"Robot module 'robot_{robot_name}' not found. "
            f"Please ensure directory 'robot_{robot_name}/' exists with file 'robot_{robot_name}.py' "
            f"containing a 'create_robot()' function."
        )


def create_core(config: dict) -> LLAGCore:
    """Create LLAGCore instance from configuration."""
    robot = create_robot_instance(config)
    
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
    
    # Print configuration summary
    print(f"=== Initializing LLAG System ===")
    print(f"Robot: {config['robot']}")
    print(f"Virtual visualization: {config['use_virtual_robot']}")
    print(f"Real robot: {config['use_real_robot']}")
    print(f"Pipeline: {'short' if config['short_pipeline'] else 'long'}")
    print(f"Modulation: {config['modulate']}")
    print(f"Debug mode: {config['debug']}")
    print()
    
    # Create core from configuration
    core = create_core(config)
    
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

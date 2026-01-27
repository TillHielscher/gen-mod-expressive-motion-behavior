# gen-mod-expressive-motion-behavior

## Installation

Install required dependencies:
- animation-dmp
- pepper-toolbox

## Configuration

The system is configured via `config.yaml`. This single file controls all aspects of the system.

### Configuration Options

- `robot`: Robot name (e.g., `pepper`, `go2`, `franka`)
  - Must match the directory structure: `robot_{name}/robot_{name}.py`
- `use_real_robot`: Connect to physical robot hardware
- `use_virtual_robot`: Enable 3D visualization
- `short_pipeline`: Use faster planning pipeline
- `modulate`: Enable expressive animation modulation
- `debug`: Enable detailed logging
- `prompt_data_path`: Path to LLM prompt configuration

### Adding a New Robot

To add a new robot (e.g., "myrobot"):

1. Create directory: `robot_myrobot/`
2. Create file: `robot_myrobot/robot_myrobot.py`
3. Implement:
   - A robot class inheriting from `RobotBase`
   - A `create_robot(robot_name)` function that returns an instance
4. Add config file: `robot_myrobot/robot_data.yaml` (or `robot_myrobot.yaml`)
5. Create primitives directory: `robot_myrobot/robot_myrobot_primitives/`
6. Add motion primitives in DMP format (`.json` + `_weights.npy` pairs)
7. Set `robot: myrobot` in `config.yaml`

That's it! The system will automatically discover and load your robot.

### Available Robots

- **pepper** - Pepper humanoid robot (SoftBank Robotics)
  - Full-body gestures with arms, head, and torso
  - URDF-based visualization
  
- **gen3_lite** - Kinova Gen3 Lite 7-DOF robotic arm
  - Expressive arm gestures and manipulation
  - Viser-based visualization (no URDF needed)

### Example Configuration

```yaml
robot: pepper
use_real_robot: false
use_virtual_robot: true
short_pipeline: true
modulate: true
debug: false
prompt_data_path: "prompts_v4.yaml"
```

## Usage

Simply run:
```bash
python main.py
```

The system automatically loads your configuration and initializes the appropriate robot.

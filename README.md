# gen-mod-expressive-motion-behavior

## Installation

Install required dependencies:
- animation-dmp
- pepper-toolbox

## Configuration

The system is configured via `config.yaml`. Edit this file to customize the robot and system behavior.

### Configuration Options

#### Robot Selection
Set the `robot` field to one of: `pepper`, `go2`, `franka`, or `custom`

#### Operation Modes
- `use_real_robot`: Connect to physical robot hardware (currently Pepper only)
- `use_virtual_robot`: Enable 3D visualization of robot movements
- `short_pipeline`: Use faster planning pipeline (recommended)
- `modulate`: Enable animation modulation for expressive motion
- `debug`: Enable detailed logging

#### Paths
- `prompt_data_path`: Path to LLM prompt configuration

#### Custom Robot
When using `robot: custom`, configure:
- `urdf_path`: Path to robot URDF file
- `primitive_lib_path`: Path to motion primitive library
- `robot_description_path`: Path to robot capabilities YAML

### Example Configurations

**Default (Pepper robot with visualization):**
```yaml
robot: pepper
use_real_robot: false
use_virtual_robot: true
short_pipeline: true
modulate: true
debug: false
```

**Debug mode with real Pepper robot:**
```yaml
robot: pepper
use_real_robot: true
use_virtual_robot: true
short_pipeline: true
modulate: true
debug: true
```

**Go2 quadruped robot:**
```yaml
robot: go2
use_real_robot: false
use_virtual_robot: true
```

## Usage

Run the system with your configured settings:
```bash
python main.py
```

The system will load configuration from `config.yaml` automatically.

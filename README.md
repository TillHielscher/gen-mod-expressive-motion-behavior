# Generation and Modulation of Expressive Robot Motion Behavior


The method aligns expressive motion behavior with an open-ended, multimodal interaction context. By automating the parametrization of animation DMPs, the method combines meaningful sequence generation with context-aware expressivity modulation. The implementation decouples control from specific robot hardware, allowing application of the method across various morphologies.

This is the implementation of the paper:

> **Paper Title**
>
> Author 1, Author 2, Author 3
>
> Venue, Year
>
> [![arXiv](https://img.shields.io/badge/arXiv-pdf-red)](https://arxiv.org/abs/xxxx.xxxxx)

## Installation

### 1. Create an isolated environment

The code is tested on python 3.10, but should work on newer and older versions

```bash
conda create -n gen-mod-expressive-motion-behavior python=3.10
```

### 2. Install dependencies

First install the Animation DMP library:
```bash
git clone https://github.com/<user>/animation-dmp.git
pip install -e animation-dmp/
```

Then install the remaining dependencies:
```bash
pip install -r requirements.txt
```

### 3. Set up an foundation model backend

**Option A – OpenAI (default)**

Set your API key as an environment variable:

```bash
export OPENAI_API_KEY="sk-..."
```

To make this permanent, add the line to your shell profile (`~/.zshrc`, `~/.bashrc`, etc.).

**Option B – Ollama (local, free)**

Install Ollama following the [official instructions](https://ollama.com/download), then pull a model:

```bash
ollama pull gemma3:4b
```

Note that Ollama does **not** support audio contexts.

## Configuration

All settings live in a single file, `config.yaml`:

```yaml
# Robot selection (pepper | gen3_lite | go2)
robot: pepper

# Operation modes
use_real_robot: false # can be implemented for individual robots
use_virtual_robot: true

# Planning
short_pipeline: false        # true = single LLM call, false = three-stage pipeline
modulate: true              # Apply animation-principle modulation to DMP playback

# LLM backend ("openai" or "ollama")
llm_backend: "openai"

# OpenAI model ID
openai_model: "gpt-4.1"

# Ollama model ID and host
ollama_model: "gemma3:4b"
ollama_host: "http://localhost:11434"

# Paths
prompt_data_path: "prompts.yaml"
```

## Usage

```bash
python main.py
```

The system reads `config.yaml`, loads the selected robot, and starts the core.
A Viser 3D viewer opens at **http://localhost:8088**.

### Providing context

Context can be entered in two ways:

1. **Terminal** – Type a text prompt and press Enter.
2. **Web GUI** – Use the text, image path, and audio path fields in the Viser sidebar.

The system accepts any combination of modalities:

| Modality | Input |
|----------|-------|
| Text | A string describing the situation |
| Image | File path to an image |
| Audio | File path to an audio file |

Modalities can be combined freely (e.g. text + image). When context is submitted, the planner generates a motion sequence and the robot begins playback. Animation principle sliders in the GUI allow real-time adjustment of the planned sequence.

## Adding a New Robot

### Currently supported (virtual) robots

| Robot | Type | DOF | Description |
|-------|------|-----|-------------|
| `pepper` | Humanoid | 17 | Arms, head, and torso gestures (SoftBank Robotics) |
| `gen3_lite` | Robot arm | 7 | Kinova Gen3 Lite 7-DOF manipulator |
| `go2` | Quadruped | 12 | Unitree Go2 with floating-base visualisation |

Each robot lives in its own directory:

```
robot_{name}/
├── robot_{name}.py                 # Robot class (extends RobotBase)
├── robot_{name}.yaml               # Capabilities, primitive library, parameter ranges
├── robot_{name}_description/       # URDF and meshes
├── robot_{name}_primitives/        # DMP files (.json + _weights.npy)
└── convert_trajectories.py         # (optional) Convert raw trajectories to DMPs
```

Steps:

1. Create the directory structure above.
2. Implement a class inheriting from `RobotBase` (`session.py`):
   - **Required** – `get_joint_names()`, `get_joint_index()`, `get_urdf_path()`, `execute_state_on_virtual_robot()`
   - **Recommended** – `get_initial_joint_angles()` (default standing pose)
   - **Optional** – `handle_rt()` (real-time modulation), `compute_base_transform()` (floating-base robots)
3. Add a `create_robot(robot_name)` factory function in the same file.
4. Create a YAML config with `capabilities`, `primitive_lib`, and `parameter_ranges`.
5. Record trajectories, convert them to DMPs, and place them in the primitives directory.
6. Set `robot: {name}` in `config.yaml`.



## Citation

```bibtex
@inproceedings{authorYEAR,
  title     = {Paper Title},
  author    = {Author 1 and Author 2 and Author 3},
  booktitle = {Venue},
  year      = {YEAR}
}
```

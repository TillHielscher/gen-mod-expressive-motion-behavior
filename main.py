"""
Main entry point.

Reads ``config.yaml``, dynamically loads the robot module, and starts the Core.
"""

from __future__ import annotations

import asyncio
import importlib
from pathlib import Path

import yaml

from core import Core


def load_config(path: str = "config.yaml") -> dict:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def create_robot(config: dict):
    """Dynamically import ``robot_{name}.robot_{name}`` and call its ``create_robot()``."""
    name = config["robot"]
    module_name = f"robot_{name}.robot_{name}"
    try:
        mod = importlib.import_module(module_name)
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            f"Robot module '{module_name}' not found.  "
            f"Ensure robot_{name}/robot_{name}.py exists with a create_robot() function."
        )
    if not hasattr(mod, "create_robot"):
        raise AttributeError(f"'{module_name}' has no create_robot() function.")
    return mod.create_robot(name)


def main() -> None:
    config = load_config()
    robot = create_robot(config)

    print(f"=== gen-mod-expressive-motion-behavior System ===")
    print(f"Robot        : {config['robot']}")
    print(f"Virtual      : {config['use_virtual_robot']}")
    print(f"Real robot   : {config['use_real_robot']}")
    print(f"Pipeline     : {'short' if config['short_pipeline'] else 'long'}")
    print(f"Modulation   : {config['modulate']}")
    print(f"LLM backend  : {config.get('llm_backend', 'openai')}")
    print(f"Ctrl loop    : {config.get('controller_loop_hz', 60)} Hz")
    print(f"RT loop      : {config.get('rt_data_loop_hz', 20)} Hz")
    print(f"Viser port   : {config.get('viser_port', 8088)}")
    print(f"LLM temp     : {config.get('llm_temperature', 0.0)}")
    print(f"Debug        : {config['debug']}")
    print()

    core = Core(
        robot,
        use_real_robot=config["use_real_robot"],
        use_virtual_robot=config["use_virtual_robot"],
        short_pipeline=config["short_pipeline"],
        modulate=config["modulate"],
        prompt_data_path=config["prompt_data_path"],
        llm_backend=config.get("llm_backend", "openai"),
        openai_model=config.get("openai_model", "gpt-4.1"),
        ollama_model=config.get("ollama_model", "gemma3:4b"),
        ollama_host=config.get("ollama_host", "http://localhost:11434"),
        debug=config["debug"],
        controller_loop_hz=config.get("controller_loop_hz", 60),
        rt_data_loop_hz=config.get("rt_data_loop_hz", 20),
        planner_trigger_interval=config.get("planner_trigger_interval", 0.1),
        context_listener_interval=config.get("context_listener_interval", 0.5),
        viser_port=config.get("viser_port", 8088),
        llm_temperature=config.get("llm_temperature", 0.0),
        whisper_model=config.get("whisper_model", "gpt-4o-transcribe"),
    )

    try:
        asyncio.run(core.run())
    except KeyboardInterrupt:
        print("\nShutting down.")


if __name__ == "__main__":
    main()

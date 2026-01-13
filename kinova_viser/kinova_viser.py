from __future__ import annotations

import time
from typing import Literal

import numpy as np
import tyro
from robot_descriptions.loaders.yourdfpy import load_robot_description

import viser
from viser.extras import ViserUrdf


def create_robot_control_sliders(
    server: viser.ViserServer, viser_urdf: ViserUrdf
) -> tuple[list[viser.GuiInputHandle[float]], list[float]]:
    slider_handles: list[viser.GuiInputHandle[float]] = []
    initial_config: list[float] = []
    for joint_name, (
        lower,
        upper,
    ) in viser_urdf.get_actuated_joint_limits().items():
        lower = lower if lower is not None else -np.pi
        upper = upper if upper is not None else np.pi
        initial_pos = 0.0 if lower < -0.1 and upper > 0.1 else (lower + upper) / 2.0
        slider = server.gui.add_slider(
            label=joint_name,
            min=lower,
            max=upper,
            step=1e-3,
            initial_value=initial_pos,
        )
        slider.on_update(
            lambda _: viser_urdf.update_cfg(
                np.array([slider.value for slider in slider_handles])
            )
        )
        slider_handles.append(slider)
        initial_config.append(initial_pos)
    return slider_handles, initial_config


def main(
    robot_type: Literal[
        "gen3_lite",  # ✅ Added Kinova Gen3 Lite
        "panda",
        "ur10",
        "cassie",
        "allegro_hand",
        "barrett_hand",
        "robotiq_2f85",
        "atlas_drc",
        "g1",
        "h1",
        "anymal_c",
        "go2",
    ] = "gen3_lite",  # ✅ Default is Gen3 Lite now
    load_meshes: bool = True,
    load_collision_meshes: bool = False,
) -> None:
    # Start viser server.
    server = viser.ViserServer()

    # Load URDF.
    urdf = load_robot_description(
        robot_type + "_description",  # e.g., "gen3_lite_description"
        load_meshes=load_meshes,
        build_scene_graph=load_meshes,
        load_collision_meshes=load_collision_meshes,
        build_collision_scene_graph=load_collision_meshes,
    )
    viser_urdf = ViserUrdf(
        server,
        urdf_or_path=urdf,
        load_meshes=load_meshes,
        load_collision_meshes=load_collision_meshes,
        collision_mesh_color_override=(1.0, 0.0, 0.0, 0.5),
    )

    # Create sliders for joint control.
    with server.gui.add_folder("Joint position control"):
        (slider_handles, initial_config) = create_robot_control_sliders(
            server, viser_urdf
        )

    # ✅ Override with custom initial configuration
    custom_init = [0.0, -0.6, 1.9, -1.6, 2.0, 0.0, 0.5]
    if len(custom_init) == len(slider_handles):
        initial_config = custom_init
        for s, q in zip(slider_handles, initial_config):
            s.value = q
        viser_urdf.update_cfg(np.array(initial_config))

    # Add visibility checkboxes.
    with server.gui.add_folder("Visibility"):
        show_meshes_cb = server.gui.add_checkbox(
            "Show meshes",
            viser_urdf.show_visual,
        )
        show_collision_meshes_cb = server.gui.add_checkbox(
            "Show collision meshes", viser_urdf.show_collision
        )

    @show_meshes_cb.on_update
    def _(_):
        viser_urdf.show_visual = show_meshes_cb.value

    @show_collision_meshes_cb.on_update
    def _(_):
        viser_urdf.show_collision = show_collision_meshes_cb.value

    show_meshes_cb.visible = load_meshes
    show_collision_meshes_cb.visible = load_collision_meshes

    # Set initial robot configuration.
    viser_urdf.update_cfg(np.array(initial_config))

    # Add grid under robot.
    trimesh_scene = viser_urdf._urdf.scene or viser_urdf._urdf.collision_scene
    server.scene.add_grid(
        "/grid",
        width=2,
        height=2,
        position=(
            0.0,
            0.0,
            trimesh_scene.bounds[0, 2] if trimesh_scene is not None else 0.0,
        ),
    )

    # Reset button.
    reset_button = server.gui.add_button("Reset")

    @reset_button.on_click
    def _(_):
        for s, init_q in zip(slider_handles, initial_config):
            s.value = init_q

        # -------------------------------------------
    # Keyframe recording
    # -------------------------------------------
    keyframes: list[list[float]] = []

    # Button to record keyframes
    record_button = server.gui.add_button("Record keyframe")

    @record_button.on_click
    def _(_):
        current_cfg = [s.value for s in slider_handles]
        keyframes.append(current_cfg)
        print(f"[Recorder] Recorded keyframe #{len(keyframes)}: {current_cfg}")

    # Button to save keyframes
    save_button = server.gui.add_button("Save keyframes")

    @save_button.on_click
    def _(_):
        if not keyframes:
            print("[Recorder] No keyframes to save.")
            return
        arr = np.array(keyframes)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"keyframes_{timestamp}.txt"
        np.savetxt(filename, arr, fmt="%.6f")
        print(f"[Recorder] Saved {len(keyframes)} keyframes to {filename}")


    # Keep alive.
    while True:
        time.sleep(10.0)


if __name__ == "__main__":
    tyro.cli(main)

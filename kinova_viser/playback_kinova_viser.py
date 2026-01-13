from __future__ import annotations

import time
from pathlib import Path
import threading

import numpy as np
from robot_descriptions.loaders.yourdfpy import load_robot_description

import viser
from viser.extras import ViserUrdf

import time


save_traj_txt=True


# -------------------------------
# Configuration variables
# -------------------------------
keyframes_file = Path("keyframes_20250926-191637.txt")  # change to your file
hz = 120.0
#durations = np.ones(13)
robot_type = "gen3_lite"
load_meshes = True
load_collision_meshes = False

# Start robot in this configuration
custom_init = [0.0, -0.6, 1.9, -1.6, 2.0, 0.0, 0.5]


def interpolate_keyframes(keyframes: np.ndarray, durations: np.ndarray, hz: float = 60.0) -> np.ndarray:
    """Linear interpolation between keyframes according to durations."""
    N, J = keyframes.shape
    interpolated_list = []
    for i in range(N - 1):
        start = keyframes[i]
        end = keyframes[i + 1]
        steps = max(int(durations[i] * hz), 1)
        for s in range(steps):
            alpha = s / steps
            interpolated_list.append((1 - alpha) * start + alpha * end)
    interpolated_list.append(keyframes[-1])
    return np.array(interpolated_list)


def main():
    #durations = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.5])  # seconds between consecutive keyframes
    durations = np.array([1.5, 2, 1.5])
    #durations = np.array([0])
    # -------------------------------
    # Start Viser
    # -------------------------------
    server = viser.ViserServer()

    # Load URDF
    urdf = load_robot_description(
        robot_type + "_description",
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

    # -------------------------------
    # Set initial robot configuration
    # -------------------------------
    if len(custom_init) != len(viser_urdf.get_actuated_joint_limits()):
        raise ValueError("custom_init length does not match number of actuated joints")
    viser_urdf.update_cfg(np.array(custom_init))

    # -------------------------------
    # Load keyframes
    # -------------------------------
    keyframes = np.loadtxt(keyframes_file)
    if keyframes.ndim == 1:
        keyframes = keyframes.reshape(1, -1)


    if durations.size != keyframes.shape[0] - 1:
        print(f"auto setting durations, as its wrong size: Durations length ({durations.size}) must be one less than number of keyframes ({keyframes.shape[0]})")
        durations=np.ones(keyframes.shape[0] - 1)

    from scipy.interpolate import CubicSpline

    # -------------------------------
    # Cubic spline smoothing
    # -------------------------------
    # Compute cumulative time for each keyframe
    times = np.concatenate([[0], np.cumsum(durations)])  # shape (N_keyframes,)

    N_joints = keyframes.shape[1]

    # Interpolate at desired frequency
    t_interp = np.linspace(0, times[-1], int(times[-1] * hz))

    # Create cubic spline per joint
    interpolated = np.zeros((len(t_interp), N_joints))
    for j in range(N_joints):
        cs = CubicSpline(times, keyframes[:, j])
        interpolated[:, j] = cs(t_interp)

    # Optional additional smoothing with Savitzky-Golay filter
    from scipy.signal import savgol_filter
    interpolated = savgol_filter(interpolated, window_length=11, polyorder=5, axis=0)

    if save_traj_txt:
        # Optional: timestamped filename
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_file = f"trajectory_60hz_{timestamp}.txt"
        np.savetxt(output_file, interpolated, fmt="%.6f")
        print(f"[Trajectory] Saved {interpolated.shape[0]} steps to {output_file}")

    # Interpolate in advance
    #interpolated = interpolate_keyframes(keyframes, durations, hz)

    # -------------------------------
    # GUI: Start Playback Button
    # -------------------------------
    start_button = server.gui.add_button("Start Playback")

    def play_animation():
        dt = 1.0 / hz
        for idx, q in enumerate(interpolated):
            viser_urdf.update_cfg(q)
            time.sleep(dt)

    @start_button.on_click
    def _(_):
        # Run playback in a separate thread so GUI remains responsive
        threading.Thread(target=play_animation, daemon=True).start()

    # Keep GUI alive
    while True:
        time.sleep(10.0)


if __name__ == "__main__":
    main()

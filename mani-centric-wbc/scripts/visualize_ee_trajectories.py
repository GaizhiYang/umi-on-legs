#!/usr/bin/env python3
"""Visualize end-effector pose trajectories stored in a pickle dataset."""

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.spatial.transform import Rotation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize ee_pos and ee_axis_angle trajectories from a pickle file."
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        default="data/pushing.pkl",
        type=Path,
        help="Pickle dataset path (default: data/pushing.pkl)",
    )
    parser.add_argument(
        "--episodes",
        nargs="+",
        type=int,
        default=[0],
        help="Episode indices to display (default: 0)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Display all episodes, subject to --max-trajectories",
    )
    parser.add_argument(
        "--max-trajectories",
        type=int,
        default=200,
        help="Maximum number of trajectories displayed with --all (default: 200)",
    )
    parser.add_argument(
        "--sample-step",
        type=int,
        default=1,
        help="Plot every Nth sample to reduce rendering cost (default: 1)",
    )
    parser.add_argument(
        "--show-orientation",
        action="store_true",
        help="Draw sampled end-effector coordinate frames",
    )
    parser.add_argument(
        "--orientation-step",
        type=int,
        default=250,
        help="Samples between coordinate frames (default: 250)",
    )
    parser.add_argument(
        "--orientation-scale",
        type=float,
        default=None,
        help="Coordinate-frame axis length in meters (default: automatic)",
    )
    parser.add_argument("--animate", action="store_true", help="Animate one episode")
    parser.add_argument(
        "--animation-step",
        type=int,
        default=10,
        help="Samples advanced per animation frame (default: 10)",
    )
    parser.add_argument("--fps", type=int, default=30, help="Animation FPS")
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Save the static figure, for example trajectory.png",
    )
    parser.add_argument(
        "--animation-output",
        type=Path,
        default=None,
        help="Save animation as .gif or .mp4",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open an interactive window",
    )
    return parser.parse_args()


def load_dataset(path: Path) -> List[Dict[str, np.ndarray]]:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset does not exist: {path}")
    with path.open("rb") as file:
        episodes = pickle.load(file)
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("Dataset must be a non-empty list of episode dictionaries")

    for index, episode in enumerate(episodes):
        if not isinstance(episode, dict):
            raise ValueError(f"Episode {index} is not a dictionary")
        for key in ("ee_pos", "ee_axis_angle"):
            if key not in episode:
                raise ValueError(f"Episode {index} is missing {key!r}")
            value = np.asarray(episode[key])
            if value.ndim != 2 or value.shape[1] != 3:
                raise ValueError(
                    f"Episode {index} {key} must have shape (T, 3), got {value.shape}"
                )
        if len(episode["ee_pos"]) != len(episode["ee_axis_angle"]):
            raise ValueError(f"Episode {index} position/orientation lengths differ")
    return episodes


def select_indices(args: argparse.Namespace, episode_count: int) -> List[int]:
    if args.all:
        indices = list(range(min(episode_count, args.max_trajectories)))
    else:
        indices = args.episodes
    invalid = [index for index in indices if index < 0 or index >= episode_count]
    if invalid:
        raise IndexError(
            f"Episode indices out of range: {invalid}; dataset has {episode_count} episodes"
        )
    return list(dict.fromkeys(indices))


def episode_time(episode: Dict[str, np.ndarray], length: int) -> np.ndarray:
    if "t" in episode:
        time = np.asarray(episode["t"]).reshape(-1)
        if len(time) >= length:
            return time[:length]
    return np.arange(length, dtype=float) / 200.0


def set_equal_3d_limits(axis, positions: np.ndarray) -> float:
    minimum = positions.min(axis=0)
    maximum = positions.max(axis=0)
    center = (minimum + maximum) / 2.0
    radius = max(float((maximum - minimum).max()) / 2.0, 0.05)
    margin = radius * 1.1
    axis.set_xlim(center[0] - margin, center[0] + margin)
    axis.set_ylim(center[1] - margin, center[1] + margin)
    axis.set_zlim(center[2] - margin, center[2] + margin)
    axis.set_box_aspect((1, 1, 1))
    return radius


def draw_coordinate_frame(axis, position, rotation, scale: float, alpha=0.8):
    colors = ("tab:red", "tab:green", "tab:blue")
    for direction, color in zip(rotation.T, colors):
        endpoint = position + direction * scale
        axis.plot(
            [position[0], endpoint[0]],
            [position[1], endpoint[1]],
            [position[2], endpoint[2]],
            color=color,
            alpha=alpha,
            linewidth=1.2,
        )


def build_figure(
    episodes: Sequence[Dict[str, np.ndarray]],
    indices: Sequence[int],
    args: argparse.Namespace,
):
    figure = plt.figure(figsize=(14, 8), constrained_layout=True)
    grid = figure.add_gridspec(3, 2, width_ratios=(1.5, 1.0))
    axis_3d = figure.add_subplot(grid[:, 0], projection="3d")
    component_axes = [figure.add_subplot(grid[i, 1]) for i in range(3)]
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(indices)))
    all_positions = np.concatenate([np.asarray(episodes[i]["ee_pos"]) for i in indices])
    radius = set_equal_3d_limits(axis_3d, all_positions)
    frame_scale = args.orientation_scale or max(radius * 0.08, 0.03)

    for color, index in zip(colors, indices):
        episode = episodes[index]
        position = np.asarray(episode["ee_pos"])
        orientation = np.asarray(episode["ee_axis_angle"])
        time = episode_time(episode, len(position))
        samples = slice(None, None, args.sample_step)
        label = f"episode {index} ({len(position)} samples, {time[-1]:.2f} s)"
        axis_3d.plot(*position[samples].T, color=color, linewidth=1.6, label=label)
        axis_3d.scatter(*position[0], color=color, marker="o", s=25)
        axis_3d.scatter(*position[-1], color=color, marker="x", s=35)
        for component, component_axis in enumerate(component_axes):
            component_axis.plot(
                time[samples], position[samples, component], color=color, linewidth=1.2
            )

        if args.show_orientation:
            rotations = Rotation.from_rotvec(orientation).as_matrix()
            for frame_index in range(0, len(position), args.orientation_step):
                draw_coordinate_frame(
                    axis_3d,
                    position[frame_index],
                    rotations[frame_index],
                    frame_scale,
                )

    axis_3d.set_title(f"End-Effector Trajectories: {args.dataset.name}")
    axis_3d.set_xlabel("X [m]")
    axis_3d.set_ylabel("Y [m]")
    axis_3d.set_zlabel("Z [m]")
    if len(indices) <= 15:
        axis_3d.legend(loc="upper left", fontsize=8)
    axis_3d.grid(True)
    for component_axis, name in zip(component_axes, "XYZ"):
        component_axis.set_ylabel(f"{name} [m]")
        component_axis.grid(True, alpha=0.3)
    component_axes[-1].set_xlabel("Time [s]")
    figure.suptitle(
        "Circle: start    Cross: end    Frame axes: X red, Y green, Z blue",
        fontsize=10,
    )
    return figure, axis_3d, frame_scale


def animate_episode(
    figure,
    axis_3d,
    episode: Dict[str, np.ndarray],
    episode_index: int,
    frame_scale: float,
    args: argparse.Namespace,
) -> FuncAnimation:
    position = np.asarray(episode["ee_pos"])
    rotations = Rotation.from_rotvec(np.asarray(episode["ee_axis_angle"])).as_matrix()
    frame_indices = np.arange(0, len(position), args.animation_step)
    trail, = axis_3d.plot([], [], [], color="tab:orange", linewidth=2.5)
    marker, = axis_3d.plot([], [], [], marker="o", color="black", markersize=6)
    frame_lines = [axis_3d.plot([], [], [], linewidth=2)[0] for _ in range(3)]
    for line, color in zip(frame_lines, ("tab:red", "tab:green", "tab:blue")):
        line.set_color(color)

    def update(frame_number):
        index = int(frame_indices[frame_number])
        current = position[index]
        trail.set_data_3d(position[: index + 1].T)
        marker.set_data_3d([current[0]], [current[1]], [current[2]])
        for direction, line in zip(rotations[index].T, frame_lines):
            endpoint = current + direction * frame_scale
            line.set_data_3d(
                [current[0], endpoint[0]],
                [current[1], endpoint[1]],
                [current[2], endpoint[2]],
            )
        axis_3d.set_title(
            f"Episode {episode_index}: sample {index}/{len(position) - 1}"
        )
        return trail, marker, *frame_lines

    return FuncAnimation(
        figure,
        update,
        frames=len(frame_indices),
        interval=1000.0 / args.fps,
        blit=False,
        repeat=True,
    )


def save_animation(animation: FuncAnimation, output: Path, fps: int):
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".gif":
        animation.save(output, writer=PillowWriter(fps=fps))
    elif output.suffix.lower() == ".mp4":
        animation.save(output, writer="ffmpeg", fps=fps)
    else:
        raise ValueError("Animation output must end in .gif or .mp4")


def main():
    args = parse_args()
    for name in ("sample_step", "orientation_step", "animation_step", "fps"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be greater than zero")

    episodes = load_dataset(args.dataset)
    indices = select_indices(args, len(episodes))
    lengths = [len(episodes[index]["ee_pos"]) for index in indices]
    print(f"Loaded {len(episodes)} episodes from {args.dataset}")
    print(f"Displaying episodes {indices}; lengths: {lengths}")

    figure, axis_3d, frame_scale = build_figure(episodes, indices, args)
    animation = None
    if args.animate or args.animation_output is not None:
        if len(indices) != 1:
            raise ValueError("Animation requires exactly one selected episode")
        animation = animate_episode(
            figure, axis_3d, episodes[indices[0]], indices[0], frame_scale, args
        )

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.save, dpi=180)
        print(f"Saved figure to {args.save}")
    if args.animation_output is not None:
        save_animation(animation, args.animation_output, args.fps)
        print(f"Saved animation to {args.animation_output}")
    if not args.no_show:
        plt.show()
    else:
        plt.close(figure)


if __name__ == "__main__":
    main()

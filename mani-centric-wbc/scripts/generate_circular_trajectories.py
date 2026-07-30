import argparse
import pickle
from pathlib import Path

import numpy as np
import pytorch3d.transforms as p3d
import torch
from transforms3d import euler


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate circular end-effector trajectories in pickle format."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/circular.pkl"),
        help="Output pickle path (default: data/circular.pkl).",
    )
    parser.add_argument(
        "--center",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=(0.0, 0.0, 0.4),
        help="Circle center in meters (default: 0 0 0.4).",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=0.2,
        help="Circle radius in meters (default: 0.2).",
    )
    parser.add_argument(
        "--radius-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        help="Randomize each episode's radius in this range instead of --radius.",
    )
    parser.add_argument(
        "--plane",
        choices=("xy", "xz", "yz"),
        default="xy",
        help="Plane containing the circle (default: xy).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Trajectory duration in seconds (default: 10).",
    )
    parser.add_argument(
        "--sampling-rate",
        type=float,
        default=200.0,
        help="Pose sampling rate in Hz (default: 200).",
    )
    parser.add_argument(
        "--turns",
        type=float,
        default=1.0,
        help="Number of circles completed during one episode (default: 1).",
    )
    parser.add_argument(
        "--start-angle-deg",
        type=float,
        default=0.0,
        help="Initial angle in degrees (default: 0).",
    )
    parser.add_argument(
        "--clockwise",
        action="store_true",
        help="Travel clockwise instead of counterclockwise.",
    )
    parser.add_argument(
        "--num-trajectories",
        type=int,
        default=200,
        help="Number of episodes to generate (default: 200).",
    )
    parser.add_argument(
        "--random-start-angle",
        action="store_true",
        help="Add an independent uniform start-angle offset to each episode.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for radius and start-angle randomization (default: 0).",
    )
    parser.add_argument(
        "--orientation-rpy",
        type=float,
        nargs=3,
        metavar=("ROLL", "PITCH", "YAW"),
        default=(-np.pi / 2, 0.0, -np.pi / 2),
        help="Fixed end-effector RPY orientation in radians.",
    )
    return parser.parse_args()


def validate_args(args):
    if args.radius <= 0:
        raise ValueError("--radius must be positive")
    if args.radius_range is not None:
        radius_min, radius_max = args.radius_range
        if radius_min <= 0 or radius_max < radius_min:
            raise ValueError("--radius-range must satisfy 0 < MIN <= MAX")
    if args.duration <= 0:
        raise ValueError("--duration must be positive")
    if args.sampling_rate <= 0:
        raise ValueError("--sampling-rate must be positive")
    if args.turns <= 0:
        raise ValueError("--turns must be positive")
    if args.num_trajectories <= 0:
        raise ValueError("--num-trajectories must be positive")


def generate_circle(center, radius, phase, plane):
    ee_pos = np.repeat(np.asarray(center, dtype=np.float64)[None, :], len(phase), axis=0)
    cos_phase = radius * np.cos(phase)
    sin_phase = radius * np.sin(phase)

    axis_a, axis_b = {
        "xy": (0, 1),
        "xz": (0, 2),
        "yz": (1, 2),
    }[plane]
    ee_pos[:, axis_a] += cos_phase
    ee_pos[:, axis_b] += sin_phase
    return ee_pos


def main():
    args = parse_args()
    validate_args(args)

    episode_len = int(round(args.duration * args.sampling_rate))
    if episode_len < 2:
        raise ValueError("duration * sampling rate must produce at least two samples")

    rng = np.random.RandomState(args.seed)
    sample_fraction = np.arange(episode_len, dtype=np.float64) / episode_len
    t = np.arange(episode_len + 1, dtype=np.float64) / args.sampling_rate
    direction = -1.0 if args.clockwise else 1.0

    rotation = euler.euler2mat(*args.orientation_rpy, axes="sxyz")
    axis_angle = p3d.matrix_to_axis_angle(torch.from_numpy(rotation)).numpy()
    ee_axis_angle = np.repeat(axis_angle[None, :], episode_len, axis=0)
    gripper_width = np.zeros((episode_len, 3), dtype=np.float64)

    episodes = []
    radii = []
    for _ in range(args.num_trajectories):
        if args.radius_range is None:
            radius = args.radius
        else:
            radius = rng.uniform(*args.radius_range)

        start_angle = np.deg2rad(args.start_angle_deg)
        if args.random_start_angle:
            start_angle += rng.uniform(0.0, 2.0 * np.pi)

        phase = start_angle + direction * 2.0 * np.pi * args.turns * sample_fraction
        ee_pos = generate_circle(args.center, radius, phase, args.plane)
        episodes.append(
            {
                "t": t.copy(),
                "ee_pos": ee_pos,
                "ee_axis_angle": ee_axis_angle.copy(),
                "gripper_width": gripper_width.copy(),
            }
        )
        radii.append(radius)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as output_file:
        pickle.dump(episodes, output_file)

    print(f"Saved {len(episodes)} trajectories to {args.output}")
    print(
        f"Samples per trajectory: {episode_len}, duration: {args.duration:g} s, "
        f"sampling rate: {args.sampling_rate:g} Hz"
    )
    print(
        f"Center: {tuple(args.center)}, plane: {args.plane}, "
        f"radius range: [{min(radii):.4g}, {max(radii):.4g}] m"
    )


if __name__ == "__main__":
    main()

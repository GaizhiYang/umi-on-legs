"""Load a URDF in Isaac Gym for visual and kinematic debugging."""

import copy
import os
import tempfile
import xml.etree.ElementTree as ET

from isaacgym import gymapi, gymutil
import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_URDF = os.path.join(
    PROJECT_ROOT, "resources", "robots", "go2_arx5", "d1_arx5.urdf"
)

# Same pose as config/env/env_d1ARX5.yaml.
DEFAULT_DOF_POSITIONS = {
    "FL_hip_joint": 0.0,
    "FL_thigh_joint": 0.8,
    "FL_calf_joint": -1.6,
    "FL_foot_joint": 0.0,
    "FR_hip_joint": 0.0,
    "FR_thigh_joint": 0.8,
    "FR_calf_joint": -1.6,
    "FR_foot_joint": 0.0,
    "RL_hip_joint": 0.0,
    "RL_thigh_joint": 1.0,
    "RL_calf_joint": -1.6,
    "RL_foot_joint": 0.0,
    "RR_hip_joint": 0.0,
    "RR_thigh_joint": 1.0,
    "RR_calf_joint": -1.6,
    "RR_foot_joint": 0.0,
    "joint1": 0.0,
    "joint2": 0.3,
    "joint3": 0.5,
    "joint4": 0.0,
    "joint5": 0.0,
    "joint6": 0.0,
}


def parse_joint_positions(value):
    """Parse a comma-separated list such as joint1=0.2,joint2=0.5."""
    positions = {}
    if not value:
        return positions
    for item in value.split(","):
        try:
            name, position = item.split("=", 1)
            positions[name.strip()] = float(position)
        except ValueError as exc:
            raise ValueError(
                "Invalid --joint_positions value. Use NAME=VALUE,NAME=VALUE."
            ) from exc
    return positions


def create_collision_debug_urdf(urdf_path):
    """Create a temporary URDF that renders collision elements as visuals."""
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    num_collisions = 0

    for link in root.findall("link"):
        for visual in link.findall("visual"):
            link.remove(visual)
        for collision in link.findall("collision"):
            visual = copy.deepcopy(collision)
            visual.tag = "visual"
            visual.set("name", "collision_debug_%d" % num_collisions)
            link.append(visual)
            num_collisions += 1

    handle, debug_path = tempfile.mkstemp(
        prefix="collision_debug_",
        suffix=".urdf",
        dir=os.path.dirname(urdf_path),
    )
    os.close(handle)
    tree.write(debug_path, encoding="utf-8", xml_declaration=True)
    return debug_path, num_collisions


def print_asset_info(gym, asset):
    body_names = gym.get_asset_rigid_body_names(asset)
    joint_names = gym.get_asset_joint_names(asset)
    dof_names = gym.get_asset_dof_names(asset)
    dof_props = gym.get_asset_dof_properties(asset)

    print("\n=== Isaac Gym asset information ===")
    print(
        "Rigid bodies: %d, joints: %d, DOFs: %d"
        % (len(body_names), len(joint_names), len(dof_names))
    )
    print("\nRigid bodies:")
    for index, name in enumerate(body_names):
        print("  [%02d] %s" % (index, name))

    print("\nDOFs:")
    for index, name in enumerate(dof_names):
        print(
            "  [%02d] %-20s limits=[% .4f, % .4f]"
            % (index, name, dof_props["lower"][index], dof_props["upper"][index])
        )
    print()


def print_body_poses(gym, env, actor):
    names = gym.get_actor_rigid_body_names(env, actor)
    states = gym.get_actor_rigid_body_states(env, actor, gymapi.STATE_POS)
    print("\n=== Rigid-body world poses ===")
    for index, name in enumerate(names):
        position = states["pose"]["p"][index]
        rotation = states["pose"]["r"][index]
        print(
            "  [%02d] %-24s p=(% .4f, % .4f, % .4f) "
            "q=(% .4f, % .4f, % .4f, % .4f)"
            % (
                index,
                name,
                position["x"],
                position["y"],
                position["z"],
                rotation["x"],
                rotation["y"],
                rotation["z"],
                rotation["w"],
            )
        )
    print()


def main():
    args = gymutil.parse_arguments(
        description="Load and inspect a URDF in Isaac Gym",
        custom_parameters=[
            {
                "name": "--urdf",
                "type": str,
                "default": DEFAULT_URDF,
                "help": "Absolute or project-relative URDF path",
            },
            {
                "name": "--base_height",
                "type": float,
                "default": 0.48,
                "help": "Initial base height in metres",
            },
            {
                "name": "--joint_positions",
                "type": str,
                "default": "",
                "help": "Overrides in NAME=VALUE,NAME=VALUE form (radians)",
            },
            {
                "name": "--zero_pose",
                "action": "store_true",
                "help": "Start all DOFs at zero instead of the D1ARX5 config pose",
            },
            {
                "name": "--free_base",
                "action": "store_true",
                "help": "Do not fix the robot base",
            },
            {
                "name": "--enable_gravity",
                "action": "store_true",
                "help": "Enable gravity (disabled by default for inspection)",
            },
            {
                "name": "--no_collapse_fixed_joints",
                "action": "store_true",
                "help": "Disable fixed-joint collapsing",
            },
            {
                "name": "--no_flip_visual_attachments",
                "action": "store_true",
                "help": "Disable visual attachment flipping",
            },
            {
                "name": "--no_replace_cylinders",
                "action": "store_true",
                "help": "Do not replace collision cylinders with capsules",
            },
            {
                "name": "--show_axes",
                "action": "store_true",
                "help": "Show every rigid body's coordinate frame initially",
            },
            {
                "name": "--show_collisions",
                "action": "store_true",
                "help": "Render collision geometry instead of visual geometry",
            },
            {
                "name": "--num_steps",
                "type": int,
                "default": -1,
                "help": "Exit after this many simulation steps; -1 keeps the viewer open",
            },
            {
                "name": "--screenshot",
                "type": str,
                "default": "",
                "help": "Write the final viewer frame to this image path",
            },
        ],
    )

    urdf_path = args.urdf
    if not os.path.isabs(urdf_path):
        urdf_path = os.path.join(PROJECT_ROOT, urdf_path)
    urdf_path = os.path.abspath(urdf_path)
    if not os.path.isfile(urdf_path):
        raise FileNotFoundError("URDF does not exist: %s" % urdf_path)

    debug_urdf_path = None
    if args.show_collisions:
        debug_urdf_path, num_collisions = create_collision_debug_urdf(urdf_path)
        urdf_path = debug_urdf_path
        print("Rendering %d URDF collision elements." % num_collisions)

    gym = gymapi.acquire_gym()
    sim_params = gymapi.SimParams()
    sim_params.dt = 1.0 / 60.0
    sim_params.substeps = 2
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline
    if args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.num_threads = args.num_threads
        sim_params.physx.use_gpu = args.use_gpu

    sim = gym.create_sim(
        args.compute_device_id,
        args.graphics_device_id,
        args.physics_engine,
        sim_params,
    )
    if sim is None:
        raise RuntimeError("Failed to create Isaac Gym simulation")

    viewer = None
    try:
        viewer = gym.create_viewer(sim, gymapi.CameraProperties())
        if viewer is None:
            raise RuntimeError("Failed to create Isaac Gym viewer")

        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        gym.add_ground(sim, plane_params)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = int(gymapi.DOF_MODE_POS)
        asset_options.fix_base_link = not args.free_base
        asset_options.disable_gravity = not args.enable_gravity
        asset_options.collapse_fixed_joints = not args.no_collapse_fixed_joints
        asset_options.flip_visual_attachments = (
            False
            if args.show_collisions
            else not args.no_flip_visual_attachments
        )
        asset_options.replace_cylinder_with_capsule = not args.no_replace_cylinders
        asset_options.use_mesh_materials = True
        asset_options.thickness = 0.01

        asset_root = os.path.dirname(urdf_path)
        asset_file = os.path.basename(urdf_path)
        print("Loading URDF: %s" % urdf_path)
        print(
            "Import options: collapse_fixed_joints=%s, "
            "flip_visual_attachments=%s, replace_cylinders=%s"
            % (
                asset_options.collapse_fixed_joints,
                asset_options.flip_visual_attachments,
                asset_options.replace_cylinder_with_capsule,
            )
        )
        asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
        if asset is None:
            raise RuntimeError("Isaac Gym failed to load the URDF")
        print_asset_info(gym, asset)

        env = gym.create_env(
            sim,
            gymapi.Vec3(-1.5, -1.5, 0.0),
            gymapi.Vec3(1.5, 1.5, 1.5),
            1,
        )
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(0.0, 0.0, args.base_height)
        actor = gym.create_actor(env, asset, start_pose, "robot", 0, 0)
        if args.show_collisions:
            collision_color = gymapi.Vec3(0.95, 0.25, 0.1)
            for body_index in range(gym.get_actor_rigid_body_count(env, actor)):
                gym.set_rigid_body_color(
                    env,
                    actor,
                    body_index,
                    gymapi.MESH_VISUAL,
                    collision_color,
                )

        dof_names = gym.get_actor_dof_names(env, actor)
        dof_name_to_index = gym.get_actor_dof_dict(env, actor)
        desired_positions = {} if args.zero_pose else dict(DEFAULT_DOF_POSITIONS)
        desired_positions.update(parse_joint_positions(args.joint_positions))
        unknown_names = sorted(set(desired_positions) - set(dof_name_to_index))
        if unknown_names:
            print("Warning: DOFs not found in this URDF: %s" % ", ".join(unknown_names))

        dof_states = np.zeros(len(dof_names), dtype=gymapi.DofState.dtype)
        for name, position in desired_positions.items():
            if name in dof_name_to_index:
                dof_states["pos"][dof_name_to_index[name]] = position
        gym.set_actor_dof_states(env, actor, dof_states, gymapi.STATE_ALL)

        dof_props = gym.get_actor_dof_properties(env, actor)
        dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
        dof_props["stiffness"].fill(100.0)
        dof_props["damping"].fill(5.0)
        gym.set_actor_dof_properties(env, actor, dof_props)
        gym.set_actor_dof_position_targets(env, actor, dof_states["pos"])
        gym.prepare_sim(sim)

        gym.viewer_camera_look_at(
            viewer,
            env,
            gymapi.Vec3(1.35, 1.15, 1.0),
            gymapi.Vec3(0.0, 0.0, 0.35),
        )
        gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_A, "toggle_axes")
        gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_P, "print_poses")
        gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_R, "reset_pose")

        show_axes = args.show_axes
        axes_geometry = gymutil.AxesGeometry(0.08)
        print(
            "Viewer mode: %s"
            % ("collision geometry" if args.show_collisions else "visual geometry")
        )
        print("Viewer controls: A = toggle body axes, P = print poses, R = reset pose")

        step = 0
        while not gym.query_viewer_has_closed(viewer):
            for event in gym.query_viewer_action_events(viewer):
                if event.value <= 0:
                    continue
                if event.action == "toggle_axes":
                    show_axes = not show_axes
                elif event.action == "print_poses":
                    print_body_poses(gym, env, actor)
                elif event.action == "reset_pose":
                    gym.set_actor_dof_states(env, actor, dof_states, gymapi.STATE_ALL)
                    gym.set_actor_dof_position_targets(env, actor, dof_states["pos"])

            gym.simulate(sim)
            gym.fetch_results(sim, True)
            gym.step_graphics(sim)

            gym.clear_lines(viewer)
            if show_axes:
                body_states = gym.get_actor_rigid_body_states(
                    env, actor, gymapi.STATE_POS
                )
                for pose_buffer in body_states["pose"]:
                    pose = gymapi.Transform.from_buffer(pose_buffer)
                    gymutil.draw_lines(
                        axes_geometry, gym, viewer, env, pose
                    )

            gym.draw_viewer(viewer, sim, True)
            if args.screenshot and (
                args.num_steps >= 0 and step + 1 >= args.num_steps
            ):
                screenshot_path = os.path.abspath(args.screenshot)
                gym.write_viewer_image_to_file(viewer, screenshot_path)
                print("Screenshot: %s" % screenshot_path)
            gym.sync_frame_time(sim)
            step += 1
            if args.num_steps >= 0 and step >= args.num_steps:
                break
    finally:
        if viewer is not None:
            gym.destroy_viewer(viewer)
        gym.destroy_sim(sim)
        if debug_urdf_path is not None and os.path.exists(debug_urdf_path):
            os.remove(debug_urdf_path)


if __name__ == "__main__":
    main()

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, MultiMeshRayCasterCfg, patterns
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.classic.cartpole.mdp as mdp
import isaaclab_tasks.manager_based.ithor.mdp as ithormdp
import isaaclab_tasks.manager_based.navigation.mdp as navmdp
from isaaclab_tasks.manager_based.ithor import _ITHOR_VALID_GOAL_POSITIONS, _ITHOR_VALID_ROBOT_POSES

from isaaclab_assets import LIMO_CONFIG


def get_valid_goal_positions(room_id):
    try:
        valid_pos = _ITHOR_VALID_GOAL_POSITIONS[str(room_id)]
    except:  # noqa
        valid_pos = None
    return valid_pos


def get_valid_robot_pose(room_id):
    try:
        robot_data = _ITHOR_VALID_ROBOT_POSES[str(room_id)]
        robot_position = robot_data[0]
        robot_rotation = robot_data[1]
    except:  # noqa
        print("Defaulting robot position and orientation...")
        robot_position = [0, 0, 0]  # default
        robot_rotation = [1, 0, 0, 0]  # default
    return (robot_position, robot_rotation)


def make_scene_cfg(room_id=0, num_envs=1, env_spacing=4.0):
    @configclass
    class IthorSceneCfg(InteractiveSceneCfg):
        """Configuration for a ithor scenes."""

        terrain = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
        ground = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Scene",
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"./assets/usd/scenes/ithor/FloorPlan{room_id}_physics/scene.usda",
            ),
        )
        ## lights
        dome_light = AssetBaseCfg(
            prim_path="/World/Light",
            spawn=sim_utils.DomeLightCfg(intensity=1000.0, color=(0.75, 0.75, 0.75)),
        )

        # robot
        robot: ArticulationCfg = LIMO_CONFIG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=ArticulationCfg.InitialStateCfg(
                pos=get_valid_robot_pose(room_id)[0],
                rot=get_valid_robot_pose(room_id)[1],
                joint_pos={
                    "front_left_wheel": 0.0,
                    "front_right_wheel": 0.0,
                    "rear_left_wheel": 0.0,
                    "rear_right_wheel": 0.0,
                },
            ),
        )

        height_scanner = MultiMeshRayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/LIMO/chassis_link",
            # offset=MultiMeshRayCaster.OffsetCfg(pos=(0.0, 0.0, 20.0)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(resolution=0.08, size=[0.7, 0.7]),
            debug_vis=True,
            mesh_prim_paths=["/World/envs/env_.*/Scene"],
        )
        contact_forces = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/LIMO/.*", history_length=3, track_air_time=True
        )

    return IthorSceneCfg(num_envs=num_envs, env_spacing=env_spacing)


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_effort = mdp.FourWheeledJointVelocityActionCfg(
        asset_name="robot",
        joint_names=[
            "front_left_wheel",
            "front_right_wheel",
            "rear_left_wheel",
            "rear_right_wheel",
        ],
        scale=3.0,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        body_pose = ObsTerm(func=mdp.body_pose_w)

        pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "pose_command"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.height_scan_quantized, params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.145}
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # reset
    reset_robot_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "front_left_wheel",
                    "front_right_wheel",
                    "rear_left_wheel",
                    "rear_right_wheel",
                ],
            ),
            "position_range": (-0.0, 0.0),
            "velocity_range": (-0.0, 0.0),
        },
    )
    reset_robot_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {},  # default to 0
            "velocity_range": {},
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # (1) Constant running reward
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    # (2) Failure penalty
    # terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    position_tracking = RewTerm(
        func=ithormdp.position_command_error,
        weight=-3.0,
        params={"std": 0.2, "command_name": "pose_command"},
    )
    position_tracking_fine_grained = RewTerm(
        func=ithormdp.position_command_error_tanh,
        weight=1.0,
        params={"std": 0.2, "command_name": "pose_command"},
    )
    # orientation_tracking = RewTerm(
    #     func=ithormdp.heading_command_error_abs,
    #     weight=-1.0,
    #     params={"command_name": "pose_command"},
    # )
    collision_penalty = RewTerm(
        func=ithormdp.collision_reward,
        weight=-5.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces"),
            "threshold": 1.0,
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # base_contact = DoneTerm(
    #     func=mdp.illegal_contact,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names="chassis_link"),
    #         "threshold": 1.0,
    #     },
    # )


def make_commands_cfg(room_id):
    @configclass
    class CommandsCfg:
        """Command terms for the MDP."""

        pose_command = navmdp.GoalPositionCommandCfg(
            asset_name="robot",
            simple_heading=False,
            resampling_time_range=(15.0, 15.0),
            debug_vis=True,
            ranges=navmdp.GoalPositionCommandCfg.Ranges(
                pos_x=(-1.5, 1.5), pos_y=(-1.5, 1.5), heading=(math.pi, math.pi)
            ),
            fixed_positions=get_valid_goal_positions(room_id),
            # ranges=navmdp.UniformPose2dCommandCfg.Ranges(
            #     pos_x=(-1.0, 1.0), pos_y=(-1.0, 1.0), heading=(math.pi, math.pi)
            # ),
        )

    return CommandsCfg()


##
# Environment configuration
##


@configclass
class IthorEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the Ithor environment."""

    room_id: int = 212
    # Scene settings
    scene = None
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    commands = None
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        self.scene = make_scene_cfg(num_envs=16, env_spacing=12.0, room_id=self.room_id)
        self.commands = make_commands_cfg(room_id=self.room_id)
        # general settings
        self.decimation = 10
        self.episode_length_s = self.episode_length_s = self.commands.pose_command.resampling_time_range[1]
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 30
        self.sim.render_interval = self.decimation / 10

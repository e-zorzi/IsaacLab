# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, MultiMeshRayCasterCfg, patterns
from isaaclab.sim import schemas
from isaaclab.sim.spawners.shapes import CuboidCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.classic.cartpole.mdp as mdp
import isaaclab_tasks.manager_based.ithor.mdp as ithormdp
import isaaclab_tasks.manager_based.navigation.mdp as navmdp
from isaaclab_tasks.manager_based.ithor import _ITHOR_VALID_GOAL_POSITIONS, _ITHOR_VALID_ROBOT_POSES

from isaaclab_assets import LIMO_CONFIG

_DEBUG_RAYCASTER = False
_DEBUG_GOAL = False
##
# Scene definition
##
SCENE_NUM = 999

try:
    _VALID_GOAL_POSITIONS = _ITHOR_VALID_GOAL_POSITIONS[str(SCENE_NUM)]
except:  # noqa
    _VALID_GOAL_POSITIONS = None


try:
    robot_data = _ITHOR_VALID_ROBOT_POSES[str(SCENE_NUM)]
    robot_position = robot_data[0]
    robot_rotation = robot_data[1]
except:  # noqa
    print("Defaulting robot position and orientation...")
    robot_position = [0, 0, 0]  # default
    robot_rotation = [1, 0, 0, 0]  # default
print(robot_position, robot_rotation)


@configclass
class IthorSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # Cube 1
    cube1 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube1",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 0.5, 0.5),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=100.0),
            # Visual material (optional)
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.4, 0.4)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(1, 0, 0.33),
            rot=(1, 0, 0, 0),
        ),
    )

    # Cube 2
    cube2 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube2",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 0.5, 0.5),
            collision_props=sim_utils.CollisionPropertiesCfg(),  # Add collision
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=100.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.8, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0, 1, 0.33),
            rot=(1, 0, 0, 0),
        ),
    )

    # Cube 2
    cube3 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube3",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 0.5, 0.5),
            collision_props=sim_utils.CollisionPropertiesCfg(),  # Add collision
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=100.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.8, 0.3)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-1, 0, 0.33),
            rot=(1, 0, 0, 0),
        ),
    )

    # Cube 2
    cube4 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube4",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 0.5, 0.5),
            collision_props=sim_utils.CollisionPropertiesCfg(),  # Add collision
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=100.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.3, 0.8)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0, -1, 0.33),
            rot=(1, 0, 0, 0),
        ),
    )

    wall_north = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/WallNorth",
        spawn=sim_utils.CuboidCfg(
            size=(0.1, 6.0, 1.0),  # Thickness, Length, Height
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(3.0, 0.0, 0.5)),  # Center height is 0.5
    )

    wall_south = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/WallSouth",
        spawn=sim_utils.CuboidCfg(
            size=(0.1, 6.0, 1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-3.0, 0.0, 0.5)),
    )

    wall_east = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/WallEast",
        spawn=sim_utils.CuboidCfg(
            size=(6.0, 0.1, 1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 3.0, 0.5)),
    )

    wall_west = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/WallWest",
        spawn=sim_utils.CuboidCfg(
            size=(6.0, 0.1, 1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, -3.0, 0.5)),
    )

    roof = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Roof",
        spawn=sim_utils.CuboidCfg(
            size=(6.0, 6.0, 0.1),  # Matches the 6x6 area of the walls
            collision_props=sim_utils.CollisionPropertiesCfg(),
            # OPTIONAL: Make it slightly transparent so you can still see the robot from above
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.4, 0.4, 0.4),
                opacity=0.5,  # 0.5 makes it see-through in the editor
            ),
        ),
        # Wall height (1.0) + half of roof thickness (0.05) = 1.05
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 1.05)),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=1000.0, color=(0.75, 0.75, 0.75)),
    )

    # robot
    robot: ArticulationCfg = LIMO_CONFIG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=robot_position,
            rot=robot_rotation,
            joint_pos={
                "front_left_wheel": 0.0,
                "front_right_wheel": 0.0,
                "rear_left_wheel": 0.0,
                "rear_right_wheel": 0.0,
            },
        ),
    )

    height_scanner = MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/chassis_link",
        # offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        ray_alignment="base",
        update_period=1 / 60,
        # offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0.35, 0.0, 0.2)),
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1, vertical_fov_range=[0, 1], horizontal_fov_range=[-45, 45], horizontal_res=1.5
        ),
        # pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[0.7, 0.5]),
        debug_vis=_DEBUG_RAYCASTER,
        mesh_prim_paths=[
            "/World/envs/env_.*/WallEast",
            "/World/envs/env_.*/WallWest",
            "/World/envs/env_.*/WallNorth",
            "/World/envs/env_.*/WallSouth",
            "/World/envs/env_.*/Roof",
            "/World/envs/env_.*/Cube1",
            "/World/envs/env_.*/Cube2",
            "/World/envs/env_.*/Cube3",
            "/World/envs/env_.*/Cube4",
        ],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)


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
        # joint_pos = ObsTerm(func=mdp.joint_pos_rel)
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
        func=mdp.reset_root_state_from_list,
        mode="reset",
        params={"candidates": robot_position},
    )
    # reset_cubes_position1 = EventTerm(
    #     func=mdp.reset_root_state_from_list,
    #     mode="reset",
    #     params={"candidates": [1, 0, 0.33], "asset_cfg": SceneEntityCfg("cube1")},
    # )
    # reset_cubes_position2 = EventTerm(
    #     func=mdp.reset_root_state_from_list,
    #     mode="reset",
    #     params={"candidates": [0, 1, 0.33], "asset_cfg": SceneEntityCfg("cube2")},
    # )
    # reset_cubes_position3 = EventTerm(
    #     func=mdp.reset_root_state_from_list,
    #     mode="reset",
    #     params={"candidates": [-1, 0, 0.33], "asset_cfg": SceneEntityCfg("cube3")},
    # )
    # reset_cubes_position4 = EventTerm(
    #     func=mdp.reset_root_state_from_list,
    #     mode="reset",
    #     params={"candidates": [0, -1, 0.33], "asset_cfg": SceneEntityCfg("cube4")},
    # )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # # (1) Constant running reward
    # alive = RewTerm(func=mdp.is_alive, weight=1.0)
    # (2) Failure penalty
    # terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    position_tracking = RewTerm(
        func=ithormdp.position_command_error,
        weight=-3.0,
        params={"command_name": "pose_command"},
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
        weight=-3.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces"),
            "threshold": 1.0,
        },
    )
    # angular_velocity_penalty = RewTerm(
    #     func=ithormdp.angular_velocity_reward,
    #     weight=-1.0,
    #     params={
    #         "threshold": 0.0,
    #     },
    # )


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


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    pose_command = navmdp.GoalPositionCommandCfg(
        asset_name="robot",
        simple_heading=False,
        resampling_time_range=(15.0, 15.0),
        debug_vis=_DEBUG_GOAL,
        ranges=navmdp.GoalPositionCommandCfg.Ranges(pos_x=(-1.5, 1.5), pos_y=(-1.5, 1.5), heading=(math.pi, math.pi)),
        fixed_positions=_VALID_GOAL_POSITIONS,
        # ranges=navmdp.UniformPose2dCommandCfg.Ranges(
        #     pos_x=(-1.0, 1.0), pos_y=(-1.0, 1.0), heading=(math.pi, math.pi)
        # ),
    )


##
# Environment configuration
##


@configclass
class IthorEnvTerrainCfg(ManagerBasedRLEnvCfg):
    """Configuration for the cartpole environment."""

    # Scene settings
    scene: IthorSceneCfg = IthorSceneCfg(num_envs=16, env_spacing=7.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 10
        self.episode_length_s = self.episode_length_s = self.commands.pose_command.resampling_time_range[1]
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 30
        self.sim.render_interval = self.decimation / 10

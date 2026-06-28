# Copyright (c) 2022-2026, The Isaac Lab Project Developers and Edoardo Zorzi
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

LIMO_PATH = "./source/isaaclab_assets/isaaclab_assets/robots/LIMO.usd"

LIMO_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=LIMO_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
        activate_contact_sensors=True,
    ),
    actuators={
        "front_joints": ImplicitActuatorCfg(
            joint_names_expr=["front_left_wheel", "front_right_wheel"],
            effort_limit_sim=10.0,
            velocity_limit_sim=10.0,
            stiffness=None,  # 10000.0,
            damping=None,  # 100.0,
        ),
        "rear_joints": ImplicitActuatorCfg(
            joint_names_expr=["rear_left_wheel", "rear_right_wheel"],
            effort_limit_sim=10.0,
            velocity_limit_sim=10.0,
            stiffness=None,  # 10000.0,
            damping=None,  # 100.0,
        ),
    },
)

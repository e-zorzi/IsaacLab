# Copyright (c) 2022-2026, Edoardo Zorzi and The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##
gym.register(
    id="Ithor-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ithor_env_cfg:IthorEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)

gym.register(
    id="Ithor-Terrain-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ithor_terrain_env_cfg:IthorEnvTerrainCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_terrain_ppo_cfg.yaml",
    },
)

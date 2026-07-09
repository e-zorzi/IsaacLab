# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def position_command_error_tanh(env: ManagerBasedRLEnv, std: float, command_name: str) -> torch.Tensor:
    """Reward position tracking with tanh kernel."""
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    distance = torch.norm(des_pos_b, dim=1)
    return 1 - torch.tanh(distance / std)


def position_command_error(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Reward position tracking with tanh kernel."""
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :2]
    distance = torch.norm(des_pos_b, dim=1)
    rew = distance
    # print("===== des_pos_b | reward =====")
    # print(list(des_pos_b), rew)
    # print("===========================\n")
    return rew


def position_command_distance(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :2]
    distance = torch.norm(des_pos_b, dim=1)
    return distance.unsqueeze(dim=1)


def heading_command_error_abs(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Penalize tracking orientation error."""
    command = env.command_manager.get_command(command_name)
    heading_b = command[:, 3]
    return heading_b.abs()


def collision_reward(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float):
    sensor = env.scene.sensors[sensor_cfg.name]

    # shape: (num_envs, num_bodies, 3)
    forces = sensor.data.net_forces_w

    # print("~~~~~~~~~~~~~~", sensor.data)

    collided = (torch.linalg.norm(forces[..., :2], dim=-1) > threshold).any(dim=1)
    # print("~~~~~~~~~~~~~~", collided.float())

    return collided.float()


def angular_velocity_reward(env: ManagerBasedRLEnv, threshold: float):
    """Penalize angular velocity."""
    action = env.action_manager.action[:, 1].abs()
    # print("~~~~~~~~~~~~~~", action)
    # action = torch.where(action.abs() > 0.0, 1.0, 0.0)
    # print("~~~~~~~~~~~~~~", action)
    return action


def angular_velocity_binary_reward(env: ManagerBasedRLEnv, threshold: float):
    """Penalize angular velocity."""
    action = env.action_manager.action[:, 1].abs()
    # print("~~~~~~~~~~~~~~", action)
    action = torch.where(action.abs() > 0.0, 1.0, 0.0)
    # print("~~~~~~~~~~~~~~", action)
    return action

# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.lab_assets.aloha import ALOHA_CFG

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import sample_uniform
# from .wheeled_robots import DifferentialController
from pxr import UsdGeom
import numpy as np
import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core.utils.prims import delete_prim, create_prim, set_prim_property
from pxr import Gf
from omni.isaac.core.simulation_context import SimulationContext
from omni.isaac.lab.assets import RigidObject, RigidObjectCfg
import omni.isaac.lab.utils.math as math_utils

@configclass
class AlohaEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 54
    action_scale = 10  # [N]
    num_actions = 2
    num_observations = 12
    num_states = 0
    max_episode_length = 1024
    time = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    # robot
    robot_cfg: ArticulationCfg = ALOHA_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    cart_dof_name = "right_wheel"
    pole_dof_name = "left_wheel"
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=14.0, replicate_physics=True)
    # reset
    # reward scales
    rew_scale_alive = 1.0
    rew_scale_terminated = -2.0
from pxr import Usd, UsdGeom, Gf
import omni.usd

# from omni.isaac.orbit.assets import RigidObjectCfg
# from omni.isaac.orbit.sim.spawners.from_files.from_files_cfg import UsdFileCfg
# from omni.isaac.orbit.sim.schemas.schemas_cfg import (
#     RigidBodyPropertiesCfg,
# )

import sys, os

# ZEROHERO_ROOT_DIR = os.environ["ZEROHERO_ROOT_DIR"]

# class ObjectTableSceneCfg(InteractiveSceneCfg):
#     plate = RigidObjectCfg(
#         prim_path="/World/envs/env_.*/target",
#         init_state=RigidObjectCfg.InitialStateCfg(
#             pos=[0.5, 0, 0.055], rot=[1, 0, 0, 0]
#         ),
#         spawn=UsdFileCfg(
#             usd_path=f"/home/kit/.local/share/ov/pkg/isaac-sim-4.1.0/standalone_examples/Aloha_graph/Aloha/assets/objects/bowl.usd",
#             scale=(
#                 1,
#                 1,
#                 1,
#             ),
#             rigid_props=RigidBodyPropertiesCfg(
#                 solver_position_iteration_count=16,
#                 solver_velocity_iteration_count=0,
#                 max_angular_velocity=0,
#                 max_linear_velocity=0,
#                 max_depenetration_velocity=0,
#                 disable_gravity=False,
#             ),
#         ),
#     )

class AlohaEnv(DirectRLEnv):
    cfg: AlohaEnvCfg

    def __init__(self, cfg: AlohaEnvCfg, render_mode: str | None = None, **kwargs):
        print("init")
        super().__init__(cfg, render_mode, **kwargs)
        
        self._cart_dof_idx, _ = self.cartpole.find_joints([self.cfg.cart_dof_name, self.cfg.pole_dof_name])
        # self._other_dof_idx, _ = self.cartpole.find_joints(self.cfg.other_dof_name)
        self.action_scale = self.cfg.action_scale
        self.joint_pos = self.cartpole.data.joint_pos
        # self.controller = DifferentialController(name="simple_control", wheel_radius=0.135, wheel_base=0.336)
        self.joint_vel = self.cartpole.data.joint_vel
        self.start = False
        self.time = 0

    def _setup_scene(self):
        self.cartpole = Articulation(self.cfg.robot_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])
        # create a new xform prim for all objects to be spawned under
        prim_utils.create_prim("/World/Objects", "Xform")
        # self.plate = RigidObjectCfg(
        #     prim_path="/World/envs/env_.*/target",
        #     init_state=RigidObjectCfg.InitialStateCfg(
        #         pos=[0.5, 0, 0.055], rot=[1, 0, 0, 0]
        #     ),
        #     spawn=UsdFileCfg(
        #         usd_path=f"/home/kit/.local/share/ov/pkg/isaac-sim-4.1.0/standalone_examples/Aloha_graph/Aloha/assets/objects/bowl.usd",
        #         scale=(
        #             1,
        #             1,
        #             1,
        #         ),
        #         rigid_props=RigidBodyPropertiesCfg(
        #             solver_position_iteration_count=16,
        #             solver_velocity_iteration_count=0,
        #             max_angular_velocity=0,
        #             max_linear_velocity=0,
        #             max_depenetration_velocity=0,
        #             disable_gravity=False,
        #         ),
        #     ),
        # )
        self.table = sim_utils.UsdFileCfg(usd_path=f"/home/kit/.local/share/ov/pkg/isaac-sim-4.1.0/standalone_examples/Aloha_graph/Aloha/assets/objects/bowl.usd")
        self.target_position = self.get_target_pos()
        for i in range(0,self.num_envs):
            self.table.func(f"/World/Objects/Table{i}", self.table, translation=self.target_position[i])

            # Create separate groups called "Origin1", "Origin2", "Origin3"
        # Each group will have a robot in it
        # self.origins = [[1, 1, 0.0], [1, -1, 0.0], [-1, 1, 0.0], [-1, -1, 0.0]]
        # for i in range(0,self.num_envs):
        #     self.table.func(f"/World/Objects/Table{i}", self.table, translation=self.target_position[i])

        # Rigid Object
        # cone_cfg = RigidObjectCfg(
        #     prim_path="/World/Objects/Table.*",
        #     spawn=sim_utils.ConeCfg(
        #         rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        #         mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        #         collision_props=sim_utils.CollisionPropertiesCfg(),
        #     ),
        #     init_state=RigidObjectCfg.InitialStateCfg(),
        # )
        # self.cone_object = RigidObject(cfg=cone_cfg)

        self.scene.articulations["cartpole"] = self.cartpole

        env_pos = self.scene.env_origins,
        # self.local_env_pose = torch.stack(list(env_pos), dim=0)[0]
        # self.kitchen = sim_utils.UsdFileCfg(usd_path=f"/home/kit/.local/share/ov/pkg/isaac-sim-2023.1.1/standalone_examples/base_aloha_env/Aloha/assets/scenes/sber_kitchen/sber_kitchen_ft.usd")
        # for i in range(0,self.num_envs):
        #     self.kitchen.func(f"/World/Objects/kitchen{i}", self.kitchen, translation=self.local_env_pose[i])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _reset_idx(self, env_ids: Sequence[int] | None):
        print("_reset_idx")
        if env_ids is None:
            env_ids = self.cartpole._ALL_INDICES
        super()._reset_idx(env_ids)
        self.time = 0

        joint_pos = self.cartpole.data.default_joint_pos[env_ids]
        # joint_pos[:] += sample_uniform(
        #     self.cfg.initial_pole_angle_range[0] * math.pi,
        #     self.cfg.initial_pole_angle_range[1] * math.pi,
        #     joint_pos[:, self._pole_dof_idx].shape,
        #     joint_pos.device,
        # )
        joint_vel = self.cartpole.data.default_joint_vel[env_ids]

        default_root_state = self.cartpole.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        self.cartpole.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.cartpole.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.cartpole.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self.local_robot_pose = self.get_env_local_pose(self.cartpole.data.root_pos_w)
        self.target_position = self.get_target_pos()
        # new_translation = (self.target_position[:,0],self.target_position[:,1],self.target_position[:,2])
        # print(self.target_position)

        # # reset root state
        # root_state = self.cone_object.data.default_root_state.clone()
        # # sample a random position on a cylinder around the origins
        # root_state[:, :3] += self.scene.env_origins[env_ids]
        # root_state[:, :3] += math_utils.sample_cylinder(
        #     radius=0.1, h_range=(0.25, 0.5), size=self.cone_object.num_instances, device=self.cone_object.device
        # )
        # # write root state to simulation
        # self.cone_object.write_root_pose_to_sim(root_state[:, :7])
        # self.cone_object.write_root_velocity_to_sim(root_state[:, 7:])
            # reset buffers
        # Получаем текущий контекст симуляции
        # sim_context = SimulationContext.instance()
        # for i in range(0,self.num_envs):
        #     prim_utils.set_prim_property(f"/World/Objects/Table{i}", "xformOp:translate", new_translation)
        
        # # Уведомляем физический движок об изменениях сцены
        # sim_context.step(render=False)
            # sim_context.pause()
            # sim_context.step(render=True)
            # self.table.func(f"/World/Objects/Table{i}", self.table, translation=self.target_position)
            # self.table.init_state.pos(self.target_position[i])
            # sim_context.play()        

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        max_velocity = 1.5
        max_angular_velocity = math.pi * 0.5
        # print("self.actions", self.actions)
        
        # Клонируем действия
        action = actions.clone()

        # Преобразуем действия в векторные значения для всех агентов
        raw_forward = action[:, 0]  # действия по переднему направлению для всех агентов
        raw_angular = action[:, 1]  # действия по угловой скорости для всех агентов

        # Преобразование значений из диапазона [-1, 1] в [0, 1] для forward
        forward = (raw_forward + 1.0) / 2.0
        forward_velocity = forward * max_velocity  # масштабируем по максимальной скорости

        # Угловая скорость для всех агентов
        angular_velocity = raw_angular * max_angular_velocity

        # Обновляем действия для всех агентов
        self.actions = torch.stack([forward_velocity, angular_velocity], dim=1).to("cuda")


    def _apply_action(self) -> None:     
        # Определяем базовые параметры
        wheel_base = 0.34
        wheel_radius = 0.068

        # Инициализируем тензор для всех скоростей суставов для всех агентов
        joint_velocities = torch.zeros((self.actions.shape[0], 2), device="cuda")  # (num_envs, 2)

        # Вычисляем скорости суставов для всех агентов
        joint_velocities[:, 0] = ((2 * self.actions[:, 0]) - (self.actions[:, 1] * wheel_base)) / (2 * wheel_radius)
        joint_velocities[:, 1] = ((2 * self.actions[:, 0]) + (self.actions[:, 1] * wheel_base)) / (2 * wheel_radius)
        self.cartpole.set_joint_effort_target(joint_velocities, joint_ids=self._cart_dof_idx)  # Применяем усилия

        
    def _get_observations(self) -> dict:
        # print("get_observation")
        self.robot_lin_vel = self.cartpole.data.root_lin_vel_w
        self.robot_ang_vel = self.cartpole.data.root_ang_vel_w
        stage = get_current_stage()
        
        self.prev_local_robot_pose = self.local_robot_pose
        self.local_robot_pose = self.get_env_local_pose(self.cartpole.data.root_pos_w)
        self.robot_quat = self.cartpole.data.root_quat_w
        self.local_target_pos = self.get_env_local_pose(self.target_position)
        obs = torch.cat(
            (
                self.local_robot_pose,
                self.robot_lin_vel,
                self.robot_ang_vel,
                self.local_target_pos,
            ),
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # print("_get_rewards")
        total_reward = compute_rewards(
            self.cfg.rew_scale_alive,
            self.cfg.rew_scale_terminated,
            self.reset_terminated,
            self.local_robot_pose,
            self.local_target_pos,
            self.prev_local_robot_pose,
            self.robot_lin_vel,
            self.robot_ang_vel,
            self.time,
            self.max_episode_length
        )
        print("reward", total_reward)

        return total_reward.cuda()

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.joint_pos = self.cartpole.data.joint_pos
        self.joint_vel = self.cartpole.data.joint_vel

        # print("time:", self.episode_length_buf, self.max_episode_length)
        time_out = self.episode_length_buf >= 256#self.max_episode_length - 1
        distance = torch.norm(self.local_robot_pose - self.local_target_pos, dim=1)
        out_of_bounds = distance > 8
        dones = distance < 0.65
        # print("distance", distance)
        if torch.any(out_of_bounds) or torch.any(time_out) or torch.any(dones):
            print(dones, distance, self.target_position, out_of_bounds, time_out)
        # achive_goal = 
        return out_of_bounds | dones, time_out
    
    def get_env_local_pose(self, world_pos):
        """Compute pose in env-local coordinates"""
        # xformable = UsdGeom.Xformable(stage.GetPrimAtPath(f"/World/envs/env_{i}/Robot")),
        # world_transform = xformable.ComputeLocalToWorldTransform(0)
        # world_pos = world_transform.ExtractTranslation()
        # print("world_pose", world_pos)
        
        # for i in range(self.num_envs):
        env_pos = self.scene.env_origins,
        # print("envs position:", env_pos)
        local_pose = world_pos - torch.stack(list(env_pos), dim=0)[0]
        return local_pose
    
    def get_target_pos(self):
        env_pos = self.scene.env_origins,
        # print("envs position:", env_pos)
        target_position = torch.stack(list(env_pos), dim=0)[0]
        r = 1.5 + np.random.rand()
        alpha = np.random.rand() * 2 * np.pi
        x = r * np.cos(alpha)
        y = r * np.sin(alpha)
        x = 2
        y = 2
        target_position[:,0] += x
        target_position[:,1] += y
        return target_position

@torch.jit.script
def compute_rewards(
    rew_scale_alive: float,
    rew_scale_terminated: float,
    reset_terminated: torch.Tensor,
    local_robot_pose: torch.Tensor,
    local_target_pose: torch.Tensor,
    prev_local_robot_pose: torch.Tensor,
    robot_lin_vel: torch.Tensor,
    robot_ang_vel: torch.Tensor,
    time: int,
    max_episode_length: int,
):
    # Расстояние между роботами и целями для всех агентов
    distance = torch.norm(local_robot_pose - local_target_pose, dim=1)
    prev_distance = torch.norm(prev_local_robot_pose - local_target_pose, dim=1)

    # Разница в расстоянии между предыдущим и текущим состоянием
    distance_diff = prev_distance - distance

    # Вычисление награды для каждого агента
    # reward_distance = 1000 * distance_diff  # Награда за сокращение расстояния
    # reward_lin_vel = torch.minimum(0.5 * torch.norm(robot_lin_vel, dim=1), torch.tensor(3.0))  # Награда за линейную скорость
    # reward_ang_vel = -torch.minimum(1.5 * torch.norm(robot_ang_vel, dim=1), torch.tensor(3.0))  # Штраф за угловую скорость
    reward_distance_2 = 5 / distance
    # Общая награда для каждого агента
    total_reward = reward_distance_2

    return total_reward

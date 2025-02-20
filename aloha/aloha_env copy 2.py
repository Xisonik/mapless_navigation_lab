# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

from omni.isaac.lab_assets.aloha import ALOHA_CFG

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import sample_uniform


@configclass
class AlohaEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 5.0
    action_scale = 100.0  # [N]
    num_actions = 1
    num_observations = 2
    num_states = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # robot
    robot_cfg: ArticulationCfg = ALOHA_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    cart_dof_name = "right_wheel"
    pole_dof_name = "left_wheel"
    # other_dof_name = ["fl_castor_wheel", "fr_castor_wheel", "left_wheel", "right_wheel", 
    #                                     "rl_castor_wheel", "rr_castor_wheel", "fl_joint1", "fr_joint1", 
    #                                     "lr_joint1", "fl_wheel", "fr_wheel", "rl_wheel", 
    #                                     "rr_wheel", "fl_joint2", "fr_joint2", "lr_joint2", "rr_joint2", 
    #                                     "fl_joint3", "fr_joint3", "lr_joint3", "rr_joint3", "fl_joint4", 
    #                                     "fr_joint4", "lr_joint4", "rr_joint4", "fl_joint5", "fr_joint5", 
    #                                     "lr_joint5", "rr_joint5", "fl_joint6", "fr_joint6", "lr_joint6", 
    #                                     "rr_joint6", "fl_joint7", "fl_joint8", "fr_joint7", "fr_joint8", 
    #                                     "lr_joint7", "lr_joint8", "rr_joint7", "rr_joint8"]
    #self.controller = DifferentialController(name="simple_control", wheel_radius=0.135, wheel_base=0.336)
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # reset
    initial_pole_angle_range = [-0.25, 0.25]  # the range in which the pole angle is sampled from on reset [rad]

    # reward scales
    rew_scale_alive = 1.0
    rew_scale_terminated = -2.0
    rew_scale_pole_pos = -1.0
    rew_scale_cart_vel = -0.01
    rew_scale_pole_vel = -0.005


class AlohaEnv(DirectRLEnv):
    cfg: AlohaEnvCfg

    def __init__(self, cfg: AlohaEnvCfg, render_mode: str | None = None, **kwargs):
        print("init")
        super().__init__(cfg, render_mode, **kwargs)

        self._cart_dof_idx, _ = self.cartpole.find_joints([self.cfg.cart_dof_name, self.cfg.pole_dof_name])
        # self._other_dof_idx, _ = self.cartpole.find_joints(self.cfg.other_dof_name)
        self.action_scale = self.cfg.action_scale

        self.joint_pos = self.cartpole.data.joint_pos
        print("joint_pos", self.joint_pos)
        self.joint_vel = self.cartpole.data.joint_vel
        print("joint_vel", self.joint_vel)

    def _setup_scene(self):
        print("setup scene")
        self.cartpole = Articulation(self.cfg.robot_cfg)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])
        # add articultion to scene
        self.scene.articulations["cartpole"] = self.cartpole
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        print("_pre_physics_step")
        self.actions = self.action_scale * actions.clone()

    def _apply_action(self) -> None:
        print("_apply_action")
        print("self.actions", self.actions)
        print("self._cart_dof_idx", self._cart_dof_idx)
        #self.cartpole.set_joint_effort_target(self.actions, joint_ids=self._cart_dof_idx)
        #self.cartpole.set_joint_effort_target(self.actions, joint_ids=self._pole_dof_idx)
        
    def _get_observations(self) -> dict:
        print("_get_observations")
        obs = torch.cat(
            (
                self.joint_pos[:, self._cart_dof_idx[0]].unsqueeze(dim=1),
                self.joint_vel[:, self._cart_dof_idx[0]].unsqueeze(dim=1),
            ),
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        print("_get_rewards")
        total_reward = compute_rewards(
            self.cfg.rew_scale_alive,
            self.cfg.rew_scale_terminated,
            self.reset_terminated,
        )
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        print("_get_dones")
        self.joint_pos = self.cartpole.data.joint_pos
        self.joint_vel = self.cartpole.data.joint_vel

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        out_of_bounds = torch.any(torch.tensor([False, False]))#torch.abs(self.joint_pos[:, self._cart_dof_idx]) > self.cfg.max_cart_pos, dim=1)
        #out_of_bounds = out_of_bounds | torch.any(torch.abs(self.joint_pos[:, self._pole_dof_idx]) > math.pi / 2, dim=1)
        return out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        print("_reset_idx")
        if env_ids is None:
            env_ids = self.cartpole._ALL_INDICES
        super()._reset_idx(env_ids)

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

        # self.cartpole.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        # self.cartpole.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        # self.cartpole.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)


@torch.jit.script
def compute_rewards(
    rew_scale_alive: float,
    rew_scale_terminated: float,
    reset_terminated: torch.Tensor,
):
    total_reward = rew_scale_alive * (1.0 - reset_terminated.float())
    return total_reward

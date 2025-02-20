# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for a simple Cartpole robot."""


import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets import ArticulationCfg

##
# Configuration
##
stiffness_arm_const = 500
damping_arm_const = 500

ALOHA_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/home/kit/.local/share/ov/pkg/isaac-sim-4.1.0/standalone_examples/Aloha/assets/aloha/ALOHA_with_sensor_02.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=100.0,
            max_angular_velocity=100.0,
            max_depenetration_velocity=010.0,
            enable_gyroscopic_forces=True,
            # kinematic_enabled=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=10,
            solver_velocity_iteration_count=10,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
            fix_root_link=False,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.1), joint_pos={".*": 0.0}
    ),
    actuators={
        "cart_actuator": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit=40,
            velocity_limit=10000.0,
            stiffness={
                "left_wheel": 0.0,
                "right_wheel": 0.0,
                "fl_joint.*": stiffness_arm_const,
                "fr_joint.*": stiffness_arm_const,
                "lr_joint.*": stiffness_arm_const,
                "rr_joint.*": stiffness_arm_const,
            },
            damping={
                "left_wheel": 0.0,
                "right_wheel": 0.0,
                "fl_joint.*": damping_arm_const,
                "fr_joint.*": damping_arm_const,
                "lr_joint.*": damping_arm_const,
                "rr_joint.*": damping_arm_const,
            },
        ),
    },
)
"""Configuration for a simple Cartpole robot."""

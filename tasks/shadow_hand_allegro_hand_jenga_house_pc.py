# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
import math
from unittest import TextTestRunner
from matplotlib.pyplot import axis
import numpy as np
import os
import random
import torch
import datetime
from utils.torch_jit_utils import *
from tasks.hand_base.base_task import BaseTask
from isaacgym import gymtorch
from isaacgym import gymapi
import copy
from pointnet2_ops import pointnet2_utils


class ShadowHandAllegroHandJengaHousePC(BaseTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless, agent_index=[[[0, 1, 2, 3, 4, 5]], [[0, 1, 2, 3, 4, 5]]], is_multi_agent=False, algo='ppol'):
        self.camera_time = 0
        self.cfg = cfg
        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.agent_index = agent_index
        self.is_multi_agent = is_multi_agent

        self.randomize = self.cfg["task"]["randomize"]
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        self.dist_reward_scale = self.cfg["env"]["distRewardScale"]
        self.rot_reward_scale = self.cfg["env"]["rotRewardScale"]
        self.action_penalty_scale = self.cfg["env"]["actionPenaltyScale"]
        self.success_tolerance = self.cfg["env"]["successTolerance"]
        self.reach_goal_bonus = self.cfg["env"]["reachGoalBonus"]
        self.fall_dist = self.cfg["env"]["fallDistance"]
        self.fall_penalty = self.cfg["env"]["fallPenalty"]
        self.rot_eps = self.cfg["env"]["rotEps"]

        self.vel_obs_scale = 0.2  # scale factor of velocity based observations
        self.force_torque_obs_scale = 10.0  # scale factor of velocity based observations

        self.reset_position_noise = self.cfg["env"]["resetPositionNoise"]
        self.reset_rotation_noise = self.cfg["env"]["resetRotationNoise"]
        self.reset_dof_pos_noise = self.cfg["env"]["resetDofPosRandomInterval"]
        self.reset_dof_vel_noise = self.cfg["env"]["resetDofVelRandomInterval"]

        self.shadow_hand_dof_speed_scale = self.cfg["env"]["dofSpeedScale"]
        self.use_relative_control = self.cfg["env"]["useRelativeControl"]
        self.act_moving_average = self.cfg["env"]["actionsMovingAverage"]
        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.reset_time = self.cfg["env"].get("resetTime", -1.0)
        self.print_success_stat = self.cfg["env"]["printNumSuccesses"]
        self.max_consecutive_successes = self.cfg["env"]["maxConsecutiveSuccesses"]
        self.av_factor = self.cfg["env"].get("averFactor", 0.01)
        print("Averaging factor: ", self.av_factor)

        self.transition_scale = self.cfg["env"]["transition_scale"]
        self.orientation_scale = self.cfg["env"]["orientation_scale"]

        control_freq_inv = self.cfg["env"].get("controlFrequencyInv", 1)
        if self.reset_time > 0.0:
            self.max_episode_length = int(round(self.reset_time/(control_freq_inv * self.sim_params.dt)))
            print("Reset time: ", self.reset_time)
            print("New episode length: ", self.max_episode_length)


        self.asset_files_dict = {
            "block": "urdf/objects/cube_multicolor.urdf",
            "egg": "mjcf/open_ai_assets/hand/egg.xml",
            "pen": "mjcf/open_ai_assets/hand/pen.xml",
            "bottle_cap": "mjcf/bottle_cap/mobility.urdf",
            "table": "urdf/table/mobility.urdf"
        }

        if "asset" in self.cfg["env"]:
            self.asset_files_dict["block"] = self.cfg["env"]["asset"].get("assetFileNameBlock", self.asset_files_dict["block"])
            self.asset_files_dict["egg"] = self.cfg["env"]["asset"].get("assetFileNameEgg", self.asset_files_dict["egg"])
            self.asset_files_dict["pen"] = self.cfg["env"]["asset"].get("assetFileNamePen", self.asset_files_dict["pen"])

        # can be "openai", "full_no_vel", "full", "full_state"
        self.obs_type = self.cfg["env"]["observationType"]

        if not (self.obs_type in ["openai", "full_no_vel", "full", "full_state", 'pick_bottle', 'jenga']):
            raise Exception(
                "Unknown type of observations!\nobservationType should be one of: [openai, full_no_vel, full, full_state]")

        print("Obs type:", self.obs_type)

        self.num_obs_dict = {
            "openai": 42,
            "full_no_vel": 77,
            "full": 157,
            "full_state": 422 - 11 + 6 + 3, 
            "pick_bottle": 430, 
            "jenga": 414
        }
        self.num_hand_obs = 72 + 95 + 26 + 6 + 16 + 6
        self.up_axis = 'z'

        self.fingertips = ["robot0:ffdistal", "robot0:mfdistal", "robot0:rfdistal", "robot0:lfdistal", "robot0:thdistal"]
        self.a_fingertips = ["mf4", "pf4", "if4", "th4"] #！there's a small ball attach to these tips via a fixed joint
        self.hand_center = ["robot1:palm"]
        self.num_fingertips = len(self.fingertips) + len(self.a_fingertips)
        self.num_shadow_hand_tips = len(self.fingertips)
        self.num_allegro_hand_tips = len(self.a_fingertips)
        self.use_vel_obs = False
        self.fingertip_obs = True
        self.asymmetric_obs = self.cfg["env"]["asymmetric_observations"]

        num_states = 0
        if self.asymmetric_obs:
            num_states = 211

        self.cfg["env"]["numObservations"] = self.num_obs_dict[self.obs_type]
        self.cfg["env"]["numStates"] = num_states
        if self.is_multi_agent:
            self.num_agents = 2
            self.cfg["env"]["numActions"] = 26
            
        else:
            self.num_agents = 1
            self.cfg["env"]["numActions"] = 48  
            #! shadow hand: 20 actuators + 6 root pos,ori
            #! allegro: 16 joints + 6 root pos,ori

        self.cfg["device_type"] = device_type
        self.cfg["device_id"] = device_id
        self.cfg["headless"] = headless
        # self.cfg["numObservations"] = 500
        super().__init__(cfg=self.cfg)

        if self.viewer != None:
            cam_pos = gymapi.Vec3(1.0, 0.0, 1.0)
            cam_target = gymapi.Vec3(0.0, 0.0, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        if self.obs_type == "full_state" or self.obs_type == "pick_bottle" or self.obs_type == "jenga" or self.asymmetric_obs:
            sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
            self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, self.num_fingertips * 6)
            dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
            self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs, self.num_shadow_hand_dofs+self.num_allegro_hand_dofs)

            self.dof_force_tensor = self.dof_force_tensor[:, :40]

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.shadow_hand_default_dof_pos = torch.zeros(self.num_shadow_hand_dofs, dtype=torch.float, device=self.device)
        self.shadow_hand_default_dof_pos = to_torch([0.0, 0.0, -0,  -0,  -0,  -0, -0, -0,
                                            -0,  -0, -0,  -0,  -0,  -0, -0, -0,
                                            -0,  -0, -0,  -1.04,  1.2,  0., 0, -1.57], dtype=torch.float, device=self.device)
        self.allegro_hand_default_dof_pos = torch.zeros(self.num_allegro_hand_dofs, dtype=torch.float, device=self.device)
        # self.allegro_hand_default_dof_pos[:6] = torch.tensor([0, 0, -0, 0, 0, 0], dtype=torch.float, device=self.device)
        # self.allegro_hand_default_dof_pos[12] = torch.tensor([-2], dtype=torch.float, device=self.device)

        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.shadow_hand_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_shadow_hand_dofs]
        self.shadow_hand_dof_pos = self.shadow_hand_dof_state[..., 0]
        self.shadow_hand_dof_vel = self.shadow_hand_dof_state[..., 1]

        self.allegro_hand_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, self.num_shadow_hand_dofs:self.num_shadow_hand_dofs + self.num_allegro_hand_dofs]
        self.allegro_hand_dof_pos = self.allegro_hand_dof_state[..., 0]
        self.allegro_hand_dof_vel = self.allegro_hand_dof_state[..., 1]

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)
        self.hand_positions = self.root_state_tensor[:, 0:3]
        self.hand_orientations = self.root_state_tensor[:, 3:7]
        self.hand_linvels = self.root_state_tensor[:, 7:10]
        self.hand_angvels = self.root_state_tensor[:, 10:13]
        self.saved_root_tensor = self.root_state_tensor.clone() 

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs

        self.prev_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

        self.global_indices = torch.arange(self.num_envs * 3, dtype=torch.int32, device=self.device).view(self.num_envs, -1)
        self.x_unit_tensor = to_torch([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = to_torch([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = to_torch([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        self.reset_goal_buf = self.reset_buf.clone()
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)

        self.av_factor = to_torch(self.av_factor, dtype=torch.float, device=self.device)
        self.apply_forces = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
        self.apply_torque = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)

        self.total_successes = 0
        self.total_resets = 0

    def create_sim(self):
        self.dt = self.sim_params.dt
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, self.up_axis)

        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = "assets"
        shadow_hand_asset_file = "mjcf/open_ai_assets/hand/shadow_hand_right.xml"
        allegro_hand_asset_file = "urdf/allegro_hand_model/urdf/allegro_hand_r.urdf"
        door_asset_file = "urdf/house/door/mobility.urdf"
        refrigerator_asset_file = "urdf/house/refrigerator/mobility.urdf"
        window_asset_file = "urdf/house/window2/mobility.urdf"
        
        # camera
        self.camera_props = gymapi.CameraProperties()
        self.camera_props.width = 1024
        self.camera_props.height = 1024
        self.camera_props.enable_tensors = True
        self.cameras = []
        self.camera_tensor_list_depth = []
        self.camera_tensor_list_color = []
        self.camera_view_matrix_inv_list = []
        self.camera_proj_matrix_list = []
        self.camera_u = torch.arange(0, self.camera_props.width, device=self.device)
        self.camera_v = torch.arange(0, self.camera_props.height, device=self.device)
        self.camera_v2, self.camera_u2 = torch.meshgrid(self.camera_v, self.camera_u, indexing='ij')
        
        self.env_origin = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)

        # load shadow hand asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = False
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.angular_damping = 100
        asset_options.linear_damping = 100

        allegro_asset_options = gymapi.AssetOptions()
        allegro_asset_options.flip_visual_attachments = False
        allegro_asset_options.fix_base_link = False
        allegro_asset_options.collapse_fixed_joints = True
        allegro_asset_options.disable_gravity = True
        allegro_asset_options.thickness = 0.001
        allegro_asset_options.angular_damping = 100
        allegro_asset_options.linear_damping = 100
        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True
            allegro_asset_options.use_physx_armature = True
        allegro_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
            
        shadow_hand_asset = self.gym.load_asset(self.sim, asset_root, shadow_hand_asset_file, asset_options)
        allegro_hand_asset = self.gym.load_asset(self.sim, asset_root, allegro_hand_asset_file, allegro_asset_options)

        self.num_shadow_hand_bodies = self.gym.get_asset_rigid_body_count(shadow_hand_asset)
        self.num_shadow_hand_shapes = self.gym.get_asset_rigid_shape_count(shadow_hand_asset)
        self.num_shadow_hand_dofs = self.gym.get_asset_dof_count(shadow_hand_asset)
        self.num_shadow_hand_actuators = self.gym.get_asset_actuator_count(shadow_hand_asset)
        self.num_shadow_hand_tendons = self.gym.get_asset_tendon_count(shadow_hand_asset)

        print("self.num_shadow_hand_bodies: ", self.num_shadow_hand_bodies)
        print("self.num_shadow_hand_shapes: ", self.num_shadow_hand_shapes)
        print("self.num_shadow_hand_dofs: ", self.num_shadow_hand_dofs)
        print("self.num_shadow_hand_actuators: ", self.num_shadow_hand_actuators)
        print("self.num_shadow_hand_tendons: ", self.num_shadow_hand_tendons)
        print('----------------------------------------')
        self.num_allegro_hand_bodies = self.gym.get_asset_rigid_body_count(allegro_hand_asset)
        self.num_allegro_hand_shapes = self.gym.get_asset_rigid_shape_count(allegro_hand_asset)
        self.num_allegro_hand_dofs = self.gym.get_asset_dof_count(allegro_hand_asset)
        self.num_allegro_hand_actuators = self.gym.get_asset_actuator_count(allegro_hand_asset)
        self.num_allegro_hand_tendons = self.gym.get_asset_tendon_count(allegro_hand_asset)

        print("self.num_allegro_hand_bodies: ", self.num_allegro_hand_bodies)
        print("self.num_allegro_hand_shapes: ", self.num_allegro_hand_shapes)
        print("self.num_allegro_hand_dofs: ", self.num_allegro_hand_dofs)
        print("self.num_allegro_hand_actuators: ", self.num_allegro_hand_actuators)
        print("self.num_allegro_hand_tendons: ", self.num_allegro_hand_tendons)

        # tendon set up
        limit_stiffness = 30
        t_damping = 0.1
        relevant_tendons = ["robot0:T_FFJ1c", "robot0:T_MFJ1c", "robot0:T_RFJ1c", "robot0:T_LFJ1c"]
        a_relevant_tendons = [] #！ allegro hand has no tendon
        tendon_props = self.gym.get_asset_tendon_properties(shadow_hand_asset)
        a_tendon_props = self.gym.get_asset_tendon_properties(allegro_hand_asset)

        for i in range(self.num_shadow_hand_tendons):
            for rt in relevant_tendons:
                if self.gym.get_asset_tendon_name(shadow_hand_asset, i) == rt:
                    tendon_props[i].limit_stiffness = limit_stiffness
                    tendon_props[i].damping = t_damping
            for rt in a_relevant_tendons:
                if self.gym.get_asset_tendon_name(allegro_hand_asset, i) == rt:
                    a_tendon_props[i].limit_stiffness = limit_stiffness
                    a_tendon_props[i].damping = t_damping
        self.gym.set_asset_tendon_properties(shadow_hand_asset, tendon_props)
        self.gym.set_asset_tendon_properties(allegro_hand_asset, a_tendon_props)
        
        actuated_dof_names = [self.gym.get_asset_actuator_joint_name(shadow_hand_asset, i) for i in range(self.num_shadow_hand_actuators)]
        self.shadow_hand_actuated_dof_indices = [self.gym.find_asset_dof_index(shadow_hand_asset, name) for name in actuated_dof_names]
        self.allegro_hand_actuated_dof_indices = [i for i in range(16)]

        # set shadow_hand dof properties
        shadow_hand_dof_props = self.gym.get_asset_dof_properties(shadow_hand_asset)
        allegro_hand_dof_props = self.gym.get_asset_dof_properties(allegro_hand_asset)
        
        self.shadow_hand_dof_lower_limits = []
        self.shadow_hand_dof_upper_limits = []
        self.shadow_hand_dof_default_pos = []
        self.shadow_hand_dof_default_vel = []
        self.sensors = []
        sensor_pose = gymapi.Transform()

        for i in range(self.num_shadow_hand_dofs):
            self.shadow_hand_dof_lower_limits.append(shadow_hand_dof_props['lower'][i])
            self.shadow_hand_dof_upper_limits.append(shadow_hand_dof_props['upper'][i])
            self.shadow_hand_dof_default_pos.append(0.0)
            self.shadow_hand_dof_default_vel.append(0.0)

        self.shadow_hand_actuated_dof_indices = to_torch(self.shadow_hand_actuated_dof_indices, dtype=torch.long, device=self.device)
        self.shadow_hand_dof_lower_limits = to_torch(self.shadow_hand_dof_lower_limits, device=self.device)
        self.shadow_hand_dof_upper_limits = to_torch(self.shadow_hand_dof_upper_limits, device=self.device)
        self.shadow_hand_dof_default_pos = to_torch(self.shadow_hand_dof_default_pos, device=self.device)
        self.shadow_hand_dof_default_vel = to_torch(self.shadow_hand_dof_default_vel, device=self.device)

        # set allegro_hand dof properties
        self.allegro_hand_dof_lower_limits = []
        self.allegro_hand_dof_upper_limits = []
        self.allegro_hand_dof_default_pos = []
        self.allegro_hand_dof_default_vel = []
        self.sensors = []
        sensor_pose = gymapi.Transform()

        for i in range(self.num_allegro_hand_dofs):
            self.allegro_hand_dof_lower_limits.append(allegro_hand_dof_props['lower'][i])
            self.allegro_hand_dof_upper_limits.append(allegro_hand_dof_props['upper'][i])
            self.allegro_hand_dof_default_pos.append(0.0)
            self.allegro_hand_dof_default_vel.append(0.0)
        for i in range(self.num_allegro_hand_dofs):
            allegro_hand_dof_props['stiffness'][i] = 3
            allegro_hand_dof_props['damping'][i] = 0.1
            allegro_hand_dof_props['effort'][i] = 0.5


        self.allegro_hand_actuated_dof_indices = to_torch(self.allegro_hand_actuated_dof_indices, dtype=torch.long, device=self.device)
        self.allegro_hand_dof_lower_limits = to_torch(self.allegro_hand_dof_lower_limits, device=self.device)
        self.allegro_hand_dof_upper_limits = to_torch(self.allegro_hand_dof_upper_limits, device=self.device)
        self.allegro_hand_dof_default_pos = to_torch(self.allegro_hand_dof_default_pos, device=self.device)
        self.allegro_hand_dof_default_vel = to_torch(self.allegro_hand_dof_default_vel, device=self.device)
        # load manipulated object and goal assets
        moving_object_asset_options = gymapi.AssetOptions()
        moving_object_asset_options.density = 500

        block_asset = self.gym.create_box(self.sim, 0.26, 0.04, 0.04, moving_object_asset_options)
        target_asset = self.gym.create_box(self.sim, 0.22, 0.04, 0.04, moving_object_asset_options)
        
        # create table asset
        table_dims = gymapi.Vec3(0.3, 0.3, 0.1)
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.flip_visual_attachments = False
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        table_file_path = self.asset_files_dict["table"]
        table_asset = self.gym.load_asset(self.sim, asset_root, table_file_path, asset_options)
        ground_asset = self.gym.create_box(self.sim, 2.0, 3.0, 0.02, asset_options)
        wall1_asset = self.gym.create_box(self.sim, 2.0, 0.02, 2, asset_options)
        wall2_asset = self.gym.create_box(self.sim, 0.02, 3.0, 2, asset_options)
        door_asset = self.gym.load_asset(self.sim, asset_root, door_asset_file, asset_options)
        refrigerator_asset = self.gym.load_asset(self.sim, asset_root, refrigerator_asset_file, asset_options)
        window_asset = self.gym.load_asset(self.sim, asset_root, window_asset_file, asset_options)

        self.num_object_bodies = self.gym.get_asset_rigid_body_count(block_asset)
        self.num_table_bodies = self.gym.get_asset_rigid_body_count(table_asset)
        self.num_ground_bodies = self.gym.get_asset_rigid_body_count(ground_asset)
        self.num_wall1_bodies = self.gym.get_asset_rigid_body_count(wall1_asset)
        self.num_wall2_bodies = self.gym.get_asset_rigid_body_count(wall2_asset)
        self.num_door_bodies = self.gym.get_asset_rigid_body_count(door_asset)
        self.num_window_bodies = self.gym.get_asset_rigid_body_count(window_asset)
        self.num_object_shapes = self.gym.get_asset_rigid_shape_count(block_asset)
        self.num_table_shapes = self.gym.get_asset_rigid_shape_count(table_asset)
        self.num_ground_shapes = self.gym.get_asset_rigid_shape_count(ground_asset)
        self.num_wall1_shapes = self.gym.get_asset_rigid_shape_count(wall1_asset)
        self.num_wall2_shapes = self.gym.get_asset_rigid_shape_count(wall2_asset)
        self.num_door_shapes = self.gym.get_asset_rigid_shape_count(door_asset)
        self.num_window_shapes = self.gym.get_asset_rigid_shape_count(window_asset)

        # region: init position
        shadow_hand_start_pose = gymapi.Transform()
        shadow_hand_start_pose.p = gymapi.Vec3(0, -1.04, 0.29)
        shadow_hand_start_pose.r = gymapi.Quat(-1,0,0,1)*gymapi.Quat(math.sin(math.radians(-20)),0,0,math.cos(math.radians(10)))

        allegro_hand_start_pose = gymapi.Transform()
        allegro_hand_start_pose.p = gymapi.Vec3(-0.01, -0.36, 0.66)
        allegro_hand_start_pose.r = gymapi.Quat(-1,0,0,1)*gymapi.Quat(0,1,0,0)*gymapi.Quat(0,0,math.sin(math.radians(45)),math.cos(math.radians(45)))
        
        object_start_pose1_1 = gymapi.Transform()
        object_start_pose1_1.p = gymapi.Vec3(-0.07, -0.6, 0.02+0.4367)
        object_start_pose1_1.r = gymapi.Quat(0,0,1,1)
        
        object_start_pose1_2 = gymapi.Transform()
        object_start_pose1_2.p = gymapi.Vec3(0.07, -0.6, 0.02+0.4367)
        object_start_pose1_2.r = gymapi.Quat(0,0,1,1)
        
        object_start_pose1_3 = gymapi.Transform()
        object_start_pose1_3.p = gymapi.Vec3(0, -0.6+0.07, 0.06+0.4367)
        object_start_pose1_3.r = gymapi.Quat(0,0,0,1)
        
        object_start_pose1_4 = gymapi.Transform()
        object_start_pose1_4.p = gymapi.Vec3(0, -0.6-0.07, 0.06+0.4367)
        object_start_pose1_4.r = gymapi.Quat(0,0,0,1)
    
        object_start_pose2_1 = gymapi.Transform()
        object_start_pose2_1.p = gymapi.Vec3(-0.07, -0.6, 0.1+0.4367)
        object_start_pose2_1.r = gymapi.Quat(0,0,1,1)
        
        object_start_pose2_2 = gymapi.Transform()
        object_start_pose2_2.p = gymapi.Vec3(0.07, -0.6, 0.1+0.4367)
        object_start_pose2_2.r = gymapi.Quat(0,0,1,1)
        
        object_start_pose2_3 = gymapi.Transform()
        object_start_pose2_3.p = gymapi.Vec3(0, -0.6+0.07, 0.14+0.4367)
        object_start_pose2_3.r = gymapi.Quat(0,0,0,1)
        
        object_start_pose2_4 = gymapi.Transform()
        object_start_pose2_4.p = gymapi.Vec3(0, -0.6-0.07, 0.14+0.4367)
        object_start_pose2_4.r = gymapi.Quat(0,0,0,1)
    
        object_start_pose3_1 = gymapi.Transform()
        object_start_pose3_1.p = gymapi.Vec3(-0.07, -0.6, 0.18+0.4367)
        object_start_pose3_1.r = gymapi.Quat(0,0,1,1)
        
        object_start_pose3_2 = gymapi.Transform()
        object_start_pose3_2.p = gymapi.Vec3(0.07, -0.6, 0.18+0.4367)
        object_start_pose3_2.r = gymapi.Quat(0,0,1,1)
        
        object_start_pose3_3 = gymapi.Transform()
        object_start_pose3_3.p = gymapi.Vec3(0, -0.6+0.07, 0.22+0.4367)
        object_start_pose3_3.r = gymapi.Quat(0,0,0,1)
        
        object_start_pose3_4 = gymapi.Transform()
        object_start_pose3_4.p = gymapi.Vec3(0, -0.6-0.07, 0.22+0.4367)
        object_start_pose3_4.r = gymapi.Quat(0,0,0,1)   
         
        object_start_pose4_1 = gymapi.Transform()
        object_start_pose4_1.p = gymapi.Vec3(-0.07, -0.6, 0.26+0.4367)
        object_start_pose4_1.r = gymapi.Quat(0,0,1,1)
        
        object_start_pose4_2 = gymapi.Transform()
        object_start_pose4_2.p = gymapi.Vec3(0.07, -0.6, 0.26+0.4367)
        object_start_pose4_2.r = gymapi.Quat(0,0,1,1)
        
        object_start_pose4_3 = gymapi.Transform()
        object_start_pose4_3.p = gymapi.Vec3(0, -0.6+0.07, 0.30+0.4367)
        object_start_pose4_3.r = gymapi.Quat(0,0,0,1)
        
        object_start_pose4_4 = gymapi.Transform()
        object_start_pose4_4.p = gymapi.Vec3(0, -0.6-0.07, 0.30+0.4367)
        object_start_pose4_4.r = gymapi.Quat(0,0,0,1) 
        
        target_start_pose = gymapi.Transform()
        target_start_pose.p = gymapi.Vec3(0, -0.6, 0.18+0.4367)
        target_start_pose.r = gymapi.Quat(0,0,1,1)

        ground_start_pose = gymapi.Transform()
        ground_start_pose.p = gymapi.Vec3(0, 0, 0)
        ground_start_pose.r = gymapi.Quat(0,0,0,1)

        wall1_start_pose = gymapi.Transform()
        wall1_start_pose.p = gymapi.Vec3(0, -1.5, 1)
        wall1_start_pose.r = gymapi.Quat(0,0,0,1)

        wall2_start_pose = gymapi.Transform()
        wall2_start_pose.p = gymapi.Vec3(-1, 0, 1)
        wall2_start_pose.r = gymapi.Quat(0,0,0,1)

        door_start_pose = gymapi.Transform()
        door_start_pose.p = gymapi.Vec3(-0.99, -0.2, 0.5)
        door_start_pose.r = gymapi.Quat(0,0,0,1)

        window_start_pose = gymapi.Transform()
        window_start_pose.p = gymapi.Vec3(-0.1, -1.48, 0.8)
        window_start_pose.r = gymapi.Quat(0,0,1,1)

        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(0.0, -0.6, 0.28)
        table_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        
        self.stable_pos1_1 = torch.tensor([-0.07, -0.60,  0.2557+0.201], device=self.device)
        self.stable_pos1_2 = torch.tensor([ 0.07, -0.60,  0.2557+0.201], device=self.device)
        self.stable_pos1_3 = torch.tensor([ 0.00, -0.53,  0.2956+0.201], device=self.device)
        self.stable_pos1_4 = torch.tensor([ 0.00, -0.67,  0.2956+0.201], device=self.device)
        self.stable_pos2_1 = torch.tensor([-0.07, -0.60,  0.3356+0.201], device=self.device)
        self.stable_pos2_2 = torch.tensor([ 0.07, -0.60,  0.3356+0.201], device=self.device)
        self.stable_pos2_3 = torch.tensor([-0.00, -0.53,  0.3755+0.201], device=self.device)
        self.stable_pos2_4 = torch.tensor([ 0.00, -0.67,  0.3755+0.201], device=self.device)
        self.stable_pos3_1 = torch.tensor([-0.07, -0.60,  0.4155+0.201], device=self.device)
        self.stable_pos3_2 = torch.tensor([ 0.07, -0.60,  0.4155+0.201], device=self.device)
        self.stable_pos3_3 = torch.tensor([ 0.00, -0.53,  0.4554+0.201], device=self.device)
        self.stable_pos3_4 = torch.tensor([ 0.00, -0.67,  0.4554+0.201], device=self.device)
        self.stable_pos4_1 = torch.tensor([-0.07, -0.60,  0.4954+0.201], device=self.device)
        self.stable_pos4_2 = torch.tensor([ 0.07, -0.60,  0.4955+0.201], device=self.device)
        self.stable_pos4_3 = torch.tensor([ 0.00, -0.53,  0.5354+0.201], device=self.device)
        self.stable_pos4_4 = torch.tensor([ 0.00, -0.67,  0.5354+0.201], device=self.device)

        # compute aggregate size
        max_agg_bodies = self.num_shadow_hand_bodies * 2 + 17 * self.num_object_bodies + \
                self.num_table_bodies + self.num_ground_bodies+self.num_wall1_bodies + \
                self.num_wall2_bodies +self.num_door_bodies+self.num_window_bodies
        max_agg_shapes = self.num_shadow_hand_shapes * 2 + 17 * self.num_object_shapes + \
                self.num_table_shapes+self.num_ground_shapes+self.num_wall1_shapes + \
                self.num_wall2_shapes+self.num_door_shapes+self.num_window_shapes

        self.shadow_hands = []
        self.envs = []
        self.object1_1_init_state = []
        self.object1_2_init_state = []
        self.object1_3_init_state = []
        self.object1_4_init_state = []
        self.object2_1_init_state = []
        self.object2_2_init_state = []
        self.object2_3_init_state = []
        self.object2_4_init_state = []
        self.object3_1_init_state = []
        self.object3_2_init_state = []
        self.object3_3_init_state = []
        self.object3_4_init_state = []
        self.object4_1_init_state = []
        self.object4_2_init_state = []
        self.object4_3_init_state = []
        self.object4_4_init_state = []
        self.target_init_state = []
        
        self.object1_1_indices = []
        self.object1_2_indices = []
        self.object1_3_indices = []
        self.object1_4_indices = []
        self.object2_1_indices = []
        self.object2_2_indices = []
        self.object2_3_indices = []
        self.object2_4_indices = []
        self.object3_1_indices = []
        self.object3_2_indices = []
        self.object3_3_indices = []
        self.object3_4_indices = []
        self.object4_1_indices = []
        self.object4_2_indices = []
        self.object4_3_indices = []
        self.object4_4_indices = []
        self.target_indices = []
        
        self.hand_start_states = []
        self.hand_indices = []
        self.another_hand_indices = []
        self.fingertip_indices = []
        
        self.table_indices = []
        self.fingertip_handles_asset = [self.gym.find_asset_rigid_body_index(shadow_hand_asset, name) for name in self.fingertips]
        self.fingertip_another_handles_asset = [self.gym.find_asset_rigid_body_index(allegro_hand_asset, name) for name in self.a_fingertips]

# endregion

        # create fingertip force sensors, if needed
        if self.obs_type == "full_state" or self.obs_type == "pick_bottle" or self.obs_type == "jenga" or self.asymmetric_obs:
        #     #! shadow hand define force sensors in .xml file, but allegro hand not!
            sensor_pose = gymapi.Transform()
            for ft_handle in self.fingertip_handles_asset: 
                self.gym.create_asset_force_sensor(shadow_hand_asset, ft_handle, sensor_pose)
            for ft_a_handle in self.fingertip_another_handles_asset:
                self.gym.create_asset_force_sensor(allegro_hand_asset, ft_a_handle, sensor_pose)
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )
            if self.aggregate_mode >= 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)
            origin = self.gym.get_env_origin(env_ptr)
            self.env_origin[i][0] = origin.x
            self.env_origin[i][1] = origin.y
            self.env_origin[i][2] = origin.z
            # add hand - collision filter = -1 to use asset collision filters set in mjcf loader
            shadow_hand_actor = self.gym.create_actor(env_ptr, shadow_hand_asset, shadow_hand_start_pose, "hand", i, 0, 0)
            allegro_hand_actor = self.gym.create_actor(env_ptr, allegro_hand_asset, allegro_hand_start_pose, "another_hand", i, 0, 0)

            self.hand_start_states.append([shadow_hand_start_pose.p.x, shadow_hand_start_pose.p.y, shadow_hand_start_pose.p.z,
                                           shadow_hand_start_pose.r.x, shadow_hand_start_pose.r.y, shadow_hand_start_pose.r.z, shadow_hand_start_pose.r.w,
                                           0, 0, 0, 0, 0, 0])
            self.fingertip_handles_env = [self.gym.find_actor_rigid_body_index(env_ptr,shadow_hand_actor,name,gymapi.DOMAIN_SIM) for name in self.fingertips]
            self.fingertip_another_handles_env = [self.gym.find_actor_rigid_body_index(env_ptr,allegro_hand_actor,name,gymapi.DOMAIN_SIM) for name in self.a_fingertips]
            self.gym.set_actor_dof_properties(env_ptr, shadow_hand_actor, shadow_hand_dof_props)
            hand_idx = self.gym.get_actor_index(env_ptr, shadow_hand_actor, gymapi.DOMAIN_SIM)
            self.hand_indices.append(hand_idx)
            
            self.gym.set_actor_dof_properties(env_ptr, allegro_hand_actor, allegro_hand_dof_props)
            another_hand_idx = self.gym.get_actor_index(env_ptr, allegro_hand_actor, gymapi.DOMAIN_SIM)
            self.another_hand_indices.append(another_hand_idx)            

            if self.obs_type == "full_state" or self.obs_type == "pick_bottle" or self.obs_type == "jenga" or self.asymmetric_obs:
                self.gym.enable_actor_dof_force_sensors(env_ptr, shadow_hand_actor)
                self.gym.enable_actor_dof_force_sensors(env_ptr, allegro_hand_actor)
            
            hand_rigid_body_index = [[0,1,2,3], [4,5,6,7], [8,9,10,11], [12,13,14,15], [16,17,18,19,20], [21,22,23,24,25]]

            for n in self.agent_index[0]:
                colorx = 0
                colory = 0
                colorz = 0
                for m in n:
                    for o in hand_rigid_body_index[m]:
                        self.gym.set_rigid_body_color(env_ptr, shadow_hand_actor, o, gymapi.MESH_VISUAL,
                                                gymapi.Vec3(colorx, colory, colorz))

            # add object
            block_handle1_1 = self.gym.create_actor(env_ptr, block_asset, object_start_pose1_1, "object", i, 0, 0)
            block_handle1_2 = self.gym.create_actor(env_ptr, block_asset, object_start_pose1_2, "object", i, 0, 0)
            block_handle1_3 = self.gym.create_actor(env_ptr, block_asset, object_start_pose1_3, "object", i, 0, 0)
            block_handle1_4 = self.gym.create_actor(env_ptr, block_asset, object_start_pose1_4, "object", i, 0, 0)
            block_handle2_1 = self.gym.create_actor(env_ptr, block_asset, object_start_pose2_1, "object", i, 0, 0)
            block_handle2_2 = self.gym.create_actor(env_ptr, block_asset, object_start_pose2_2, "object", i, 0, 0)
            block_handle2_3 = self.gym.create_actor(env_ptr, block_asset, object_start_pose2_3, "object", i, 0, 0)
            block_handle2_4 = self.gym.create_actor(env_ptr, block_asset, object_start_pose2_4, "object", i, 0, 0)
            block_handle3_1 = self.gym.create_actor(env_ptr, block_asset, object_start_pose3_1, "object", i, 0, 0)
            block_handle3_2 = self.gym.create_actor(env_ptr, block_asset, object_start_pose3_2, "object", i, 0, 0)
            block_handle3_3 = self.gym.create_actor(env_ptr, block_asset, object_start_pose3_3, "object", i, 0, 0)
            block_handle3_4 = self.gym.create_actor(env_ptr, block_asset, object_start_pose3_4, "object", i, 0, 0)
            block_handle4_1 = self.gym.create_actor(env_ptr, block_asset, object_start_pose4_1, "object", i, 0, 0)
            block_handle4_2 = self.gym.create_actor(env_ptr, block_asset, object_start_pose4_2, "object", i, 0, 0)
            block_handle4_3 = self.gym.create_actor(env_ptr, block_asset, object_start_pose4_3, "object", i, 0, 0)
            block_handle4_4 = self.gym.create_actor(env_ptr, block_asset, object_start_pose4_4, "object", i, 0, 0)
            
            target_handle = self.gym.create_actor(env_ptr, target_asset, target_start_pose, "object", i, 0, 0)
            
            ground_handle = self.gym.create_actor(env_ptr, ground_asset, ground_start_pose, "ground", i, 0, 0)
            wall1_handle = self.gym.create_actor(env_ptr, wall1_asset, wall1_start_pose, "wall1", i, 0, 0)
            wall2_handle = self.gym.create_actor(env_ptr, wall2_asset, wall2_start_pose, "wall2", i, 0, 0)
            door_handle = self.gym.create_actor(env_ptr, door_asset, door_start_pose, "door", i, 0, 0)
            window_handle = self.gym.create_actor(env_ptr, window_asset, window_start_pose, "window", i, 0, 0)
            self.gym.set_rigid_body_color(env_ptr, block_handle1_1, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(255./255., 231./255., 200./255.))        
            self.gym.set_rigid_body_color(env_ptr, block_handle1_2, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(255./255., 231./255., 200./255.))        
            self.gym.set_rigid_body_color(env_ptr, block_handle1_3, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(255./255., 231./255., 200./255.))        
            self.gym.set_rigid_body_color(env_ptr, block_handle1_4, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(255./255., 231./255., 200./255.))        
            self.gym.set_rigid_body_color(env_ptr, block_handle2_1, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(255./255., 231./255., 200./255.))        
            self.gym.set_rigid_body_color(env_ptr, block_handle2_2, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(255./255., 231./255., 200./255.))        
            self.gym.set_rigid_body_color(env_ptr, block_handle2_3, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(255./255., 231./255., 200./255.))        
            self.gym.set_rigid_body_color(env_ptr, block_handle2_4, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(255./255., 231./255., 200./255.))        
            self.gym.set_rigid_body_color(env_ptr, block_handle3_1, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(255./255., 231./255., 200./255.))        
            self.gym.set_rigid_body_color(env_ptr, block_handle3_2, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(255./255., 231./255., 200./255.))        
            self.gym.set_rigid_body_color(env_ptr, block_handle3_3, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(255./255., 231./255., 200./255.))        
            self.gym.set_rigid_body_color(env_ptr, block_handle3_4, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(255./255., 231./255., 200./255.))        
            self.gym.set_rigid_body_color(env_ptr, block_handle4_1, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(255./255., 231./255., 200./255.))        
            self.gym.set_rigid_body_color(env_ptr, block_handle4_2, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(255./255., 231./255., 200./255.))        
            self.gym.set_rigid_body_color(env_ptr, block_handle4_3, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(255./255., 231./255., 200./255.))        
            self.gym.set_rigid_body_color(env_ptr, block_handle4_4, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(255./255., 231./255., 200./255.))        
            self.gym.set_rigid_body_color(env_ptr, target_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(132./255., 112./255., 255./255.))        
            self.gym.set_rigid_body_color(env_ptr, ground_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(139./255., 115./255., 85./255.))        
            self.gym.set_rigid_body_color(env_ptr, wall1_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(255./255., 255./255., 255./255.))        
            self.gym.set_rigid_body_color(env_ptr, wall2_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(255./255., 255./255., 255./255.))        

            self.object1_1_init_state.append([object_start_pose1_1.p.x, object_start_pose1_1.p.y, object_start_pose1_1.p.z,
                                            object_start_pose1_1.r.x, object_start_pose1_1.r.y, object_start_pose1_1.r.z, object_start_pose1_1.r.w,
                                            0, 0, 0, 0, 0, 0])
            self.object1_2_init_state.append([object_start_pose1_2.p.x, object_start_pose1_2.p.y, object_start_pose1_2.p.z,
                                            object_start_pose1_2.r.x, object_start_pose1_2.r.y, object_start_pose1_2.r.z, object_start_pose1_2.r.w,
                                            0, 0, 0, 0, 0, 0])
            self.object1_3_init_state.append([object_start_pose1_3.p.x, object_start_pose1_3.p.y, object_start_pose1_3.p.z,
                                            object_start_pose1_3.r.x, object_start_pose1_3.r.y, object_start_pose1_3.r.z, object_start_pose1_3.r.w,
                                            0, 0, 0, 0, 0, 0])
            self.object1_4_init_state.append([object_start_pose1_4.p.x, object_start_pose1_4.p.y, object_start_pose1_4.p.z,
                                           object_start_pose1_4.r.x, object_start_pose1_4.r.y, object_start_pose1_4.r.z, object_start_pose1_4.r.w,
                                           0, 0, 0, 0, 0, 0])
            self.object2_1_init_state.append([object_start_pose2_1.p.x, object_start_pose2_1.p.y, object_start_pose2_1.p.z,
                                           object_start_pose2_1.r.x, object_start_pose2_1.r.y, object_start_pose2_1.r.z, object_start_pose2_1.r.w,
                                           0, 0, 0, 0, 0, 0])
            self.object2_2_init_state.append([object_start_pose2_2.p.x, object_start_pose2_2.p.y, object_start_pose2_2.p.z,
                                           object_start_pose2_2.r.x, object_start_pose2_2.r.y, object_start_pose2_2.r.z, object_start_pose2_2.r.w,
                                           0, 0, 0, 0, 0, 0])
            self.object2_3_init_state.append([object_start_pose2_3.p.x, object_start_pose2_3.p.y, object_start_pose2_3.p.z,
                                           object_start_pose2_3.r.x, object_start_pose2_3.r.y, object_start_pose2_3.r.z, object_start_pose2_3.r.w,
                                           0, 0, 0, 0, 0, 0])
            self.object2_4_init_state.append([object_start_pose2_4.p.x, object_start_pose2_4.p.y, object_start_pose2_4.p.z,
                                           object_start_pose2_4.r.x, object_start_pose2_4.r.y, object_start_pose2_4.r.z, object_start_pose2_4.r.w,
                                           0, 0, 0, 0, 0, 0])
            self.object3_1_init_state.append([object_start_pose3_1.p.x, object_start_pose3_1.p.y, object_start_pose3_1.p.z,
                                           object_start_pose3_1.r.x, object_start_pose3_1.r.y, object_start_pose3_1.r.z, object_start_pose3_1.r.w,
                                           0, 0, 0, 0, 0, 0])
            self.object3_2_init_state.append([object_start_pose3_2.p.x, object_start_pose3_2.p.y, object_start_pose3_2.p.z,
                                           object_start_pose3_2.r.x, object_start_pose3_2.r.y, object_start_pose3_2.r.z, object_start_pose3_2.r.w,
                                           0, 0, 0, 0, 0, 0])
            self.object3_3_init_state.append([object_start_pose3_3.p.x, object_start_pose3_3.p.y, object_start_pose3_3.p.z,
                                           object_start_pose3_3.r.x, object_start_pose3_3.r.y, object_start_pose3_3.r.z, object_start_pose3_3.r.w,
                                           0, 0, 0, 0, 0, 0])
            self.object3_4_init_state.append([object_start_pose3_4.p.x, object_start_pose3_4.p.y, object_start_pose3_4.p.z,
                                           object_start_pose3_4.r.x, object_start_pose3_4.r.y, object_start_pose3_4.r.z, object_start_pose3_4.r.w,
                                           0, 0, 0, 0, 0, 0])
            self.object4_1_init_state.append([object_start_pose4_1.p.x, object_start_pose4_1.p.y, object_start_pose4_1.p.z,
                                           object_start_pose4_1.r.x, object_start_pose4_1.r.y, object_start_pose4_1.r.z, object_start_pose4_1.r.w,
                                           0, 0, 0, 0, 0, 0])
            self.object4_2_init_state.append([object_start_pose4_2.p.x, object_start_pose4_2.p.y, object_start_pose4_2.p.z,
                                           object_start_pose4_2.r.x, object_start_pose4_2.r.y, object_start_pose4_2.r.z, object_start_pose4_2.r.w,
                                           0, 0, 0, 0, 0, 0])
            self.object4_3_init_state.append([object_start_pose4_3.p.x, object_start_pose4_3.p.y, object_start_pose4_3.p.z,
                                           object_start_pose4_3.r.x, object_start_pose4_3.r.y, object_start_pose4_3.r.z, object_start_pose4_3.r.w,
                                           0, 0, 0, 0, 0, 0])
            self.object4_4_init_state.append([object_start_pose4_4.p.x, object_start_pose4_4.p.y, object_start_pose4_4.p.z,
                                           object_start_pose4_4.r.x, object_start_pose4_4.r.y, object_start_pose4_4.r.z, object_start_pose4_4.r.w,
                                           0, 0, 0, 0, 0, 0])
            self.target_init_state.append([target_start_pose.p.x, target_start_pose.p.y, target_start_pose.p.z,
                                           target_start_pose.r.x, target_start_pose.r.y, target_start_pose.r.z, target_start_pose.r.w,
                                           0, 0, 0, 0, 0, 0])

            object1_1_idx = self.gym.get_actor_index(env_ptr, block_handle1_1, gymapi.DOMAIN_SIM)
            object1_2_idx = self.gym.get_actor_index(env_ptr, block_handle1_2, gymapi.DOMAIN_SIM)
            object1_3_idx = self.gym.get_actor_index(env_ptr, block_handle1_3, gymapi.DOMAIN_SIM)
            object1_4_idx = self.gym.get_actor_index(env_ptr, block_handle1_4, gymapi.DOMAIN_SIM)
            object2_1_idx = self.gym.get_actor_index(env_ptr, block_handle2_1, gymapi.DOMAIN_SIM)
            object2_2_idx = self.gym.get_actor_index(env_ptr, block_handle2_2, gymapi.DOMAIN_SIM)
            object2_3_idx = self.gym.get_actor_index(env_ptr, block_handle2_3, gymapi.DOMAIN_SIM)
            object2_4_idx = self.gym.get_actor_index(env_ptr, block_handle2_4, gymapi.DOMAIN_SIM)
            object3_1_idx = self.gym.get_actor_index(env_ptr, block_handle3_1, gymapi.DOMAIN_SIM)
            object3_2_idx = self.gym.get_actor_index(env_ptr, block_handle3_2, gymapi.DOMAIN_SIM)
            object3_3_idx = self.gym.get_actor_index(env_ptr, block_handle3_3, gymapi.DOMAIN_SIM)
            object3_4_idx = self.gym.get_actor_index(env_ptr, block_handle3_4, gymapi.DOMAIN_SIM)
            object4_1_idx = self.gym.get_actor_index(env_ptr, block_handle4_1, gymapi.DOMAIN_SIM)
            object4_2_idx = self.gym.get_actor_index(env_ptr, block_handle4_2, gymapi.DOMAIN_SIM)
            object4_3_idx = self.gym.get_actor_index(env_ptr, block_handle4_3, gymapi.DOMAIN_SIM)
            object4_4_idx = self.gym.get_actor_index(env_ptr, block_handle4_4, gymapi.DOMAIN_SIM)
            target_idx = self.gym.get_actor_index(env_ptr, target_handle, gymapi.DOMAIN_SIM)
            self.object1_1_indices.append(object1_1_idx)
            self.object1_2_indices.append(object1_2_idx)
            self.object1_3_indices.append(object1_3_idx)
            self.object1_4_indices.append(object1_4_idx)
            self.object2_1_indices.append(object2_1_idx)
            self.object2_2_indices.append(object2_2_idx)
            self.object2_3_indices.append(object2_3_idx)
            self.object2_4_indices.append(object2_4_idx)
            self.object3_1_indices.append(object3_1_idx)
            self.object3_2_indices.append(object3_2_idx)
            self.object3_3_indices.append(object3_3_idx)
            self.object3_4_indices.append(object3_4_idx)
            self.object4_1_indices.append(object4_1_idx)
            self.object4_2_indices.append(object4_2_idx)
            self.object4_3_indices.append(object4_3_idx)
            self.object4_4_indices.append(object4_4_idx)
            self.target_indices.append(target_idx)

            # add table
            table_handle = self.gym.create_actor(env_ptr, table_asset, table_pose, "table", i, 0, 0)
            table_idx = self.gym.get_actor_index(env_ptr, table_handle, gymapi.DOMAIN_SIM)
            self.table_indices.append(table_idx)

            #set friction
            object_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, block_handle1_1)
            for object_shape_prop in object_shape_props:
                object_shape_prop.friction = 0.5
            hand_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, shadow_hand_actor)
            for hand_shape_prop in hand_shape_props:
                hand_shape_prop.friction = 10
            self.gym.set_actor_rigid_shape_properties(env_ptr, block_handle1_1, object_shape_props)
            self.gym.set_actor_rigid_shape_properties(env_ptr, block_handle1_2, object_shape_props)
            self.gym.set_actor_rigid_shape_properties(env_ptr, block_handle1_3, object_shape_props)
            self.gym.set_actor_rigid_shape_properties(env_ptr, block_handle1_4, object_shape_props)
            self.gym.set_actor_rigid_shape_properties(env_ptr, block_handle2_1, object_shape_props)
            self.gym.set_actor_rigid_shape_properties(env_ptr, block_handle2_2, object_shape_props)
            self.gym.set_actor_rigid_shape_properties(env_ptr, block_handle2_3, object_shape_props)
            self.gym.set_actor_rigid_shape_properties(env_ptr, block_handle2_4, object_shape_props)
            self.gym.set_actor_rigid_shape_properties(env_ptr, block_handle3_1, object_shape_props)
            self.gym.set_actor_rigid_shape_properties(env_ptr, block_handle3_2, object_shape_props)
            self.gym.set_actor_rigid_shape_properties(env_ptr, block_handle3_3, object_shape_props)
            self.gym.set_actor_rigid_shape_properties(env_ptr, block_handle3_4, object_shape_props)
            self.gym.set_actor_rigid_shape_properties(env_ptr, block_handle4_1, object_shape_props)
            self.gym.set_actor_rigid_shape_properties(env_ptr, block_handle4_2, object_shape_props)
            self.gym.set_actor_rigid_shape_properties(env_ptr, block_handle4_3, object_shape_props)
            self.gym.set_actor_rigid_shape_properties(env_ptr, block_handle4_4, object_shape_props)
            
            self.gym.set_actor_rigid_shape_properties(env_ptr, target_handle, object_shape_props)
            
            self.gym.set_actor_rigid_shape_properties(env_ptr, shadow_hand_actor, hand_shape_props)
            self.gym.set_actor_rigid_shape_properties(env_ptr, allegro_hand_actor, hand_shape_props)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.shadow_hands.append(shadow_hand_actor)       
            camera_handle1 = self.gym.create_camera_sensor(env_ptr, self.camera_props)
            camera_handle2 = self.gym.create_camera_sensor(env_ptr, self.camera_props)
            camera_handle3 = self.gym.create_camera_sensor(env_ptr, self.camera_props)

            # set on the front and look towards bottom
            self.gym.set_camera_location(camera_handle1, env_ptr, gymapi.Vec3(0.0, -0.6,1.3), gymapi.Vec3(1e-8, -0.6+1e-8, 0.5))
            # left 
            self.gym.set_camera_location(camera_handle2, env_ptr, gymapi.Vec3(0.0, -0.6+1,1.3), gymapi.Vec3(0.1, -0.6-1, 0.5))
            # right
            self.gym.set_camera_location(camera_handle3, env_ptr, gymapi.Vec3(0.0, -0.6-1,1.3), gymapi.Vec3(0.1, -0.6+1, 0.5))

            cam1_tensor_depth = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, camera_handle1, gymapi.IMAGE_DEPTH)
            cam2_tensor_depth = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, camera_handle2, gymapi.IMAGE_DEPTH)
            cam3_tensor_depth = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, camera_handle3, gymapi.IMAGE_DEPTH)
            torch_cam1_tensor_depth = gymtorch.wrap_tensor(cam1_tensor_depth)
            torch_cam2_tensor_depth = gymtorch.wrap_tensor(cam2_tensor_depth)
            torch_cam3_tensor_depth = gymtorch.wrap_tensor(cam3_tensor_depth)
            cam1_tensor_color = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, camera_handle1, gymapi.IMAGE_COLOR)
            cam2_tensor_color = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, camera_handle2, gymapi.IMAGE_COLOR)
            cam3_tensor_color = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, camera_handle3, gymapi.IMAGE_COLOR)
            torch_cam1_tensor_color = gymtorch.wrap_tensor(cam1_tensor_color)
            torch_cam2_tensor_color = gymtorch.wrap_tensor(cam2_tensor_color)
            torch_cam3_tensor_color = gymtorch.wrap_tensor(cam3_tensor_color)
            cam1_vinv = torch.inverse((torch.tensor(self.gym.get_camera_view_matrix(self.sim, env_ptr, camera_handle1)))).to(self.device)
            cam2_vinv = torch.inverse((torch.tensor(self.gym.get_camera_view_matrix(self.sim, env_ptr, camera_handle2)))).to(self.device)
            cam3_vinv = torch.inverse((torch.tensor(self.gym.get_camera_view_matrix(self.sim, env_ptr, camera_handle3)))).to(self.device)
            cam1_proj = torch.tensor(self.gym.get_camera_proj_matrix(self.sim, env_ptr, camera_handle1), device=self.device)
            cam2_proj = torch.tensor(self.gym.get_camera_proj_matrix(self.sim, env_ptr, camera_handle2), device=self.device)
            cam3_proj = torch.tensor(self.gym.get_camera_proj_matrix(self.sim, env_ptr, camera_handle3), device=self.device)

            per_env_camera_tensor_list_depth = [torch_cam1_tensor_depth, torch_cam2_tensor_depth, torch_cam3_tensor_depth]
            per_env_camera_tensor_list_color = [torch_cam1_tensor_color, torch_cam2_tensor_color, torch_cam3_tensor_color]

            per_env_camera_view_matrix_inv_list = [cam1_vinv, cam2_vinv, cam3_vinv]
            per_env_camera_proj_matrix_list = [cam1_proj, cam2_proj, cam3_proj]

            self.cameras.append([camera_handle1, camera_handle2, camera_handle3])
            self.camera_tensor_list_depth.append(per_env_camera_tensor_list_depth)
            self.camera_tensor_list_color.append(per_env_camera_tensor_list_color)
            self.camera_view_matrix_inv_list.append(per_env_camera_view_matrix_inv_list)
            self.camera_proj_matrix_list.append(per_env_camera_proj_matrix_list)


        self.object1_1_init_state = to_torch(self.object1_1_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.object1_2_init_state = to_torch(self.object1_2_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.object1_3_init_state = to_torch(self.object1_3_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.object1_4_init_state = to_torch(self.object1_4_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.object2_1_init_state = to_torch(self.object2_1_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.object2_2_init_state = to_torch(self.object2_2_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.object2_3_init_state = to_torch(self.object2_3_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.object2_4_init_state = to_torch(self.object2_4_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.object3_1_init_state = to_torch(self.object3_1_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.object3_2_init_state = to_torch(self.object3_2_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.object3_3_init_state = to_torch(self.object3_3_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.object3_4_init_state = to_torch(self.object3_4_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.object4_1_init_state = to_torch(self.object4_1_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.object4_2_init_state = to_torch(self.object4_2_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.object4_3_init_state = to_torch(self.object4_3_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.object4_4_init_state = to_torch(self.object4_4_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.target_init_state = to_torch(self.target_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
  
        self.hand_start_states = to_torch(self.hand_start_states, device=self.device).view(self.num_envs, 13)

        self.fingertip_handles_asset = to_torch(self.fingertip_handles_asset, dtype=torch.long, device=self.device)
        self.fingertip_another_handles_asset = to_torch(self.fingertip_another_handles_asset, dtype=torch.long, device=self.device)
        self.fingertip_handles_env = to_torch(self.fingertip_handles_env, dtype=torch.long, device=self.device)
        self.fingertip_another_handles_env = to_torch(self.fingertip_another_handles_env, dtype=torch.long, device=self.device)

        self.hand_indices = to_torch(self.hand_indices, dtype=torch.long, device=self.device)
        self.another_hand_indices = to_torch(self.another_hand_indices, dtype=torch.long, device=self.device)
        self.object1_1_indices = to_torch(self.object1_1_indices, dtype=torch.long, device=self.device)
        self.object1_2_indices = to_torch(self.object1_2_indices, dtype=torch.long, device=self.device)
        self.object1_3_indices = to_torch(self.object1_3_indices, dtype=torch.long, device=self.device)
        self.object1_4_indices = to_torch(self.object1_4_indices, dtype=torch.long, device=self.device)
        self.object2_1_indices = to_torch(self.object2_1_indices, dtype=torch.long, device=self.device)
        self.object2_2_indices = to_torch(self.object2_2_indices, dtype=torch.long, device=self.device)
        self.object2_3_indices = to_torch(self.object2_3_indices, dtype=torch.long, device=self.device)
        self.object2_4_indices = to_torch(self.object2_4_indices, dtype=torch.long, device=self.device)
        self.object3_1_indices = to_torch(self.object3_1_indices, dtype=torch.long, device=self.device)
        self.object3_2_indices = to_torch(self.object3_2_indices, dtype=torch.long, device=self.device)
        self.object3_3_indices = to_torch(self.object3_3_indices, dtype=torch.long, device=self.device)
        self.object3_4_indices = to_torch(self.object3_4_indices, dtype=torch.long, device=self.device)
        self.object4_1_indices = to_torch(self.object4_1_indices, dtype=torch.long, device=self.device)
        self.object4_2_indices = to_torch(self.object4_2_indices, dtype=torch.long, device=self.device)
        self.object4_3_indices = to_torch(self.object4_3_indices, dtype=torch.long, device=self.device)
        self.object4_4_indices = to_torch(self.object4_4_indices, dtype=torch.long, device=self.device)
        self.target_indices = to_torch(self.target_indices, dtype=torch.long, device=self.device)

        self.table_indices = to_torch(self.table_indices, dtype=torch.long, device=self.device)

    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:], self.progress_buf[:], self.successes[:], self.consecutive_successes[:] = compute_hand_reward(
            self.rew_buf, self.reset_buf, self.progress_buf, self.successes, self.consecutive_successes,
            self.max_episode_length, 
            self.target_pos, self.target_rot,  
            self.object1_1_pos, self.object1_2_pos, self.object1_3_pos, self.object1_4_pos, 
            self.object2_1_pos, self.object2_2_pos, self.object2_3_pos, self.object2_4_pos, 
            self.object3_1_pos, self.object3_2_pos, self.object3_3_pos, self.object3_4_pos, 
            self.object4_1_pos, self.object4_2_pos, self.object4_3_pos, self.object4_4_pos, 
            self.stable_pos1_1,self.stable_pos1_2,self.stable_pos1_3,self.stable_pos1_4,
            self.stable_pos2_1,self.stable_pos2_2,self.stable_pos2_3,self.stable_pos2_4, 
            self.stable_pos3_1,self.stable_pos3_2,self.stable_pos3_3,self.stable_pos3_4, 
            self.stable_pos4_1,self.stable_pos4_2,self.stable_pos4_3,self.stable_pos4_4, 
            self.right_end_pos, self.left_end_pos,
            self.left_hand_pos, self.right_hand_pos, 
            self.right_hand_ff_pos, self.right_hand_mf_pos, self.right_hand_rf_pos, self.right_hand_lf_pos, self.right_hand_th_pos, 
            self.left_hand_ff_pos, self.left_hand_mf_pos, self.left_hand_lf_pos, self.left_hand_th_pos,
            self.dist_reward_scale, self.rot_reward_scale, self.rot_eps, self.actions, self.action_penalty_scale,
            self.success_tolerance, self.fall_dist, self.fall_penalty,
            self.max_consecutive_successes, self.av_factor
        )

        self.extras['successes'] = self.successes
        self.extras['consecutive_successes'] = self.consecutive_successes

        if self.print_success_stat:
            self.total_resets = self.total_resets + self.reset_buf.sum()
            direct_average_successes = self.total_successes + self.successes.sum()
            self.total_successes = self.total_successes + (self.successes * self.reset_buf).sum()

            # The direct average shows the overall result more quickly, but slightly undershoots long term
            # policy performance.
            print("Direct average consecutive successes = {:.1f}".format(direct_average_successes/(self.total_resets + self.num_envs)))
            if self.total_resets > 0:
                print("Post-Reset average consecutive successes = {:.1f}".format(self.total_successes/self.total_resets))
    
    
    def compute_cost(self):
        actions = self.actions.clone()

        self.cost_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float)


        self.object1_1_pos = self.root_state_tensor[self.object1_1_indices, 0:3]
        self.object1_2_pos = self.root_state_tensor[self.object1_2_indices, 0:3]
        self.object1_3_pos = self.root_state_tensor[self.object1_3_indices, 0:3]
        self.object1_4_pos = self.root_state_tensor[self.object1_4_indices, 0:3]
        self.object2_1_pos = self.root_state_tensor[self.object2_1_indices, 0:3]
        self.object2_2_pos = self.root_state_tensor[self.object2_2_indices, 0:3]
        self.object2_3_pos = self.root_state_tensor[self.object2_3_indices, 0:3]
        self.object2_4_pos = self.root_state_tensor[self.object2_4_indices, 0:3]
        self.object3_1_pos = self.root_state_tensor[self.object3_1_indices, 0:3]
        self.object3_2_pos = self.root_state_tensor[self.object3_2_indices, 0:3]
        self.object3_3_pos = self.root_state_tensor[self.object3_3_indices, 0:3]
        self.object3_4_pos = self.root_state_tensor[self.object3_4_indices, 0:3]
        self.object4_1_pos = self.root_state_tensor[self.object4_1_indices, 0:3]
        self.object4_2_pos = self.root_state_tensor[self.object4_2_indices, 0:3]
        self.object4_3_pos = self.root_state_tensor[self.object4_3_indices, 0:3]
        self.object4_4_pos = self.root_state_tensor[self.object4_4_indices, 0:3]
        
        dz_1_1 = abs(self.object1_1_pos[:, 2] - self.stable_pos1_1[2])
        dz_1_2 = abs(self.object1_2_pos[:, 2] - self.stable_pos1_2[2])
        dz_1_3 = abs(self.object1_3_pos[:, 2] - self.stable_pos1_3[2])
        dz_1_4 = abs(self.object1_4_pos[:, 2] - self.stable_pos1_4[2])
        dz_2_1 = abs(self.object2_1_pos[:, 2] - self.stable_pos2_1[2])
        dz_2_2 = abs(self.object2_2_pos[:, 2] - self.stable_pos2_2[2])
        dz_2_3 = abs(self.object2_3_pos[:, 2] - self.stable_pos2_3[2])
        dz_2_4 = abs(self.object2_4_pos[:, 2] - self.stable_pos2_4[2])
        dz_3_1 = abs(self.object3_1_pos[:, 2] - self.stable_pos3_1[2])
        dz_3_2 = abs(self.object3_2_pos[:, 2] - self.stable_pos3_2[2])
        dz_3_3 = abs(self.object3_3_pos[:, 2] - self.stable_pos3_3[2])
        dz_3_4 = abs(self.object3_4_pos[:, 2] - self.stable_pos3_4[2])
        dz_4_1 = abs(self.object4_1_pos[:, 2] - self.stable_pos4_1[2])
        dz_4_2 = abs(self.object4_2_pos[:, 2] - self.stable_pos4_2[2])
        dz_4_3 = abs(self.object4_3_pos[:, 2] - self.stable_pos4_3[2])
        dz_4_4 = abs(self.object4_4_pos[:, 2] - self.stable_pos4_4[2])
  
        loss =  (dz_1_1>0.01)*(dz_1_2>0.01)*(dz_1_3>0.01)*(dz_1_4>0.01)*\
                (dz_2_1>0.01)*(dz_2_2>0.01)*(dz_2_3>0.01)*(dz_2_4>0.01)*\
                (dz_3_1>0.01)*(dz_3_2>0.01)*(dz_3_3>0.01)*(dz_3_4>0.01)*\
                (dz_4_1>0.01)*(dz_4_2>0.01)*(dz_4_3>0.01)*(dz_4_4>0.01)
                
        self.cost_buf = copy.deepcopy(loss.float())
                
        return self.cost_buf

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        if self.obs_type == "full_state" or self.obs_type == "pick_bottle" or self.obs_type == "jenga" or self.asymmetric_obs:
            self.gym.refresh_force_sensor_tensor(self.sim)
            self.gym.refresh_dof_force_tensor(self.sim)

        self.target_pose = self.root_state_tensor[self.target_indices, 0:7]
        self.target_pos = self.root_state_tensor[self.target_indices, 0:3]
        self.target_rot = self.root_state_tensor[self.target_indices, 3:7]
        self.target_linvel = self.root_state_tensor[self.target_indices, 7:10]
        self.target_angvel = self.root_state_tensor[self.target_indices, 10:13]
        self.right_middle_pos = self.target_pos + quat_apply(self.target_rot, to_torch([1, 0, 0], device=self.device).repeat(self.num_envs, 1) * 0.08)
        self.right_middle_pos = self.right_middle_pos + quat_apply(self.target_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.0)
        self.left_end_pos = self.target_pos + quat_apply(self.target_rot, to_torch([1, 0, 0], device=self.device).repeat(self.num_envs, 1) * -0.11)
        self.left_end_pos = self.left_end_pos + quat_apply(self.target_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.0)
        self.right_end_pos = self.target_pos + quat_apply(self.target_rot, to_torch([1, 0, 0], device=self.device).repeat(self.num_envs, 1) * 0.11)
        self.right_end_pos = self.right_end_pos + quat_apply(self.target_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.0)

        # self.add_debug_lines(self.envs[0], self.right_end_pos[0], self.target_rot[0])
        # self.add_debug_lines(self.envs[0], self.left_end_pos[0], self.target_rot[0])
        self.object1_1_pos = self.root_state_tensor[self.object1_1_indices, 0:3]
        self.object1_2_pos = self.root_state_tensor[self.object1_2_indices, 0:3]
        self.object1_3_pos = self.root_state_tensor[self.object1_3_indices, 0:3]
        self.object1_4_pos = self.root_state_tensor[self.object1_4_indices, 0:3]
        self.object2_1_pos = self.root_state_tensor[self.object2_1_indices, 0:3]
        self.object2_2_pos = self.root_state_tensor[self.object2_2_indices, 0:3]
        self.object2_3_pos = self.root_state_tensor[self.object2_3_indices, 0:3]
        self.object2_4_pos = self.root_state_tensor[self.object2_4_indices, 0:3]
        self.object3_1_pos = self.root_state_tensor[self.object3_1_indices, 0:3]
        self.object3_2_pos = self.root_state_tensor[self.object3_2_indices, 0:3]
        self.object3_3_pos = self.root_state_tensor[self.object3_3_indices, 0:3]
        self.object3_4_pos = self.root_state_tensor[self.object3_4_indices, 0:3]
        self.object4_1_pos = self.root_state_tensor[self.object4_1_indices, 0:3]
        self.object4_2_pos = self.root_state_tensor[self.object4_2_indices, 0:3]
        self.object4_3_pos = self.root_state_tensor[self.object4_3_indices, 0:3]
        self.object4_4_pos = self.root_state_tensor[self.object4_4_indices, 0:3]
        #! right hand is shadow hand ,left hand is allegro
        #! here is the link 'base_link' of allegro
        self.left_hand_pos = self.rigid_body_states[:, 26, 0:3]
        self.left_hand_rot = self.rigid_body_states[:, 26, 3:7]
        self.left_hand_pos = self.left_hand_pos + quat_apply(self.left_hand_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.10)
        self.left_hand_pos = self.left_hand_pos + quat_apply(self.left_hand_rot, to_torch([0, 1, 0], device=self.device).repeat(self.num_envs, 1) * -0.02)
        # self.add_debug_lines(self.envs[0], self.left_hand_pos[0], self.left_hand_rot[0])
        self.right_hand_pos = self.rigid_body_states[:, 3, 0:3]
        self.right_hand_rot = self.rigid_body_states[:, 3, 3:7]
        #! here is the link 'palm' of shadow hand
        self.right_hand_pos = self.right_hand_pos + quat_apply(self.right_hand_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.08)
        self.right_hand_pos = self.right_hand_pos + quat_apply(self.right_hand_rot, to_torch([0, 1, 0], device=self.device).repeat(self.num_envs, 1) * -0.02)
        # right hand finger
        self.right_hand_ff_pos = self.rigid_body_states[:, 7, 0:3]
        self.right_hand_ff_rot = self.rigid_body_states[:, 7, 3:7]
        self.right_hand_ff_pos = self.right_hand_ff_pos + quat_apply(self.right_hand_ff_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)
        self.right_hand_mf_pos = self.rigid_body_states[:, 11, 0:3]
        self.right_hand_mf_rot = self.rigid_body_states[:, 11, 3:7]
        self.right_hand_mf_pos = self.right_hand_mf_pos + quat_apply(self.right_hand_mf_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)
        self.right_hand_rf_pos = self.rigid_body_states[:, 15, 0:3]
        self.right_hand_rf_rot = self.rigid_body_states[:, 15, 3:7]
        self.right_hand_rf_pos = self.right_hand_rf_pos + quat_apply(self.right_hand_rf_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)
        self.right_hand_lf_pos = self.rigid_body_states[:, 20, 0:3]
        self.right_hand_lf_rot = self.rigid_body_states[:, 20, 3:7]
        self.right_hand_lf_pos = self.right_hand_lf_pos + quat_apply(self.right_hand_lf_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)
        self.right_hand_th_pos = self.rigid_body_states[:, 25, 0:3]
        self.right_hand_th_rot = self.rigid_body_states[:, 25, 3:7]
        self.right_hand_th_pos = self.right_hand_th_pos + quat_apply(self.right_hand_th_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)
        
        # left hand finger
        self.left_hand_ff_pos = self.rigid_body_states[:, 26+4, 0:3]
        self.left_hand_ff_rot = self.rigid_body_states[:, 26+4, 3:7]
        self.left_hand_ff_pos = self.left_hand_ff_pos + quat_apply(self.left_hand_ff_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)
        self.left_hand_mf_pos = self.rigid_body_states[:, 26+8, 0:3]
        self.left_hand_mf_rot = self.rigid_body_states[:, 26+8, 3:7]
        self.left_hand_mf_pos = self.left_hand_mf_pos + quat_apply(self.left_hand_mf_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)
        self.left_hand_lf_pos = self.rigid_body_states[:, 26+12, 0:3]
        self.left_hand_lf_rot = self.rigid_body_states[:, 26+12, 3:7]
        self.left_hand_lf_pos = self.left_hand_lf_pos + quat_apply(self.left_hand_lf_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)
        self.left_hand_th_pos = self.rigid_body_states[:, 26+16, 0:3]
        self.left_hand_th_rot = self.rigid_body_states[:, 26+16, 3:7]
        self.left_hand_th_pos = self.left_hand_th_pos + quat_apply(self.left_hand_th_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)
        
        # self.fingertip_state = self.rigid_body_states[:, self.fingertip_handles_env][:, :, 0:13]
        # self.fingertip_another_state = self.rigid_body_states[:, self.fingertip_another_handles_env][:, :, 0:13]
        self.compute_full_state()

        if self.asymmetric_obs:
            self.compute_full_state(True)

    def compute_full_state(self, asymm_obs=False):

        num_ft_states_shadow = 13 * self.num_shadow_hand_tips  # 65
        num_ft_force_torques_shadow = 6 * self.num_shadow_hand_tips  # 30
        
        self.obs_buf[:, 0:self.num_shadow_hand_dofs] = unscale(self.shadow_hand_dof_pos,
                                                            self.shadow_hand_dof_lower_limits, self.shadow_hand_dof_upper_limits)

        self.obs_buf[:, self.num_shadow_hand_dofs:2*self.num_shadow_hand_dofs] = self.vel_obs_scale * self.shadow_hand_dof_vel
        self.obs_buf[:, 2*self.num_shadow_hand_dofs:3*self.num_shadow_hand_dofs] = self.force_torque_obs_scale * self.dof_force_tensor[:, :24]

        fingertip_obs_start = 72  # 24*3
        # self.obs_buf[:, fingertip_obs_start:fingertip_obs_start + num_ft_states_shadow] = self.fingertip_state.reshape(self.num_envs, num_ft_states_shadow)
        self.obs_buf[:, fingertip_obs_start + num_ft_states_shadow:fingertip_obs_start + num_ft_states_shadow +
                    num_ft_force_torques_shadow] = self.force_torque_obs_scale * self.vec_sensor_tensor[:, :30]
        
        hand_pose_start = fingertip_obs_start + 95
        self.obs_buf[:, hand_pose_start:hand_pose_start + 3] = self.right_hand_pos
        self.obs_buf[:, hand_pose_start+3:hand_pose_start+4] = get_euler_xyz(self.hand_orientations[self.hand_indices, :])[0].unsqueeze(-1)
        self.obs_buf[:, hand_pose_start+4:hand_pose_start+5] = get_euler_xyz(self.hand_orientations[self.hand_indices, :])[1].unsqueeze(-1)
        self.obs_buf[:, hand_pose_start+5:hand_pose_start+6] = get_euler_xyz(self.hand_orientations[self.hand_indices, :])[2].unsqueeze(-1)

        action_obs_start = hand_pose_start + 6
        self.obs_buf[:, action_obs_start:action_obs_start + 26] = self.actions[:, :26]

        # another_hand
        num_ft_states_allegro = 13 * self.num_allegro_hand_tips  # 52
        num_ft_force_torques_allegro = 6 * self.num_allegro_hand_tips  # 24
        another_hand_start = action_obs_start + 26
        self.obs_buf[:, another_hand_start:self.num_allegro_hand_dofs + another_hand_start] = unscale(self.allegro_hand_dof_pos,
                                                            self.allegro_hand_dof_lower_limits, self.allegro_hand_dof_upper_limits)
        self.obs_buf[:, self.num_allegro_hand_dofs + another_hand_start:2*self.num_allegro_hand_dofs + another_hand_start] = self.vel_obs_scale * self.allegro_hand_dof_vel
        self.obs_buf[:, 2*self.num_allegro_hand_dofs + another_hand_start:3*self.num_allegro_hand_dofs + another_hand_start] = self.force_torque_obs_scale * self.dof_force_tensor[:, 24:48]

        fingertip_another_obs_start = another_hand_start + 48
        # self.obs_buf[:, fingertip_another_obs_start:fingertip_another_obs_start + num_ft_states_allegro] = self.fingertip_another_state.reshape(self.num_envs, num_ft_states_allegro)
        self.obs_buf[:, fingertip_another_obs_start + num_ft_states_allegro:fingertip_another_obs_start + num_ft_states_allegro +
                    num_ft_force_torques_allegro] = self.force_torque_obs_scale * self.vec_sensor_tensor[:, 30:]

        hand_another_pose_start = fingertip_another_obs_start + 76
        self.obs_buf[:, hand_another_pose_start:hand_another_pose_start + 3] = self.left_hand_pos
        self.obs_buf[:, hand_another_pose_start+3:hand_another_pose_start+4] = get_euler_xyz(self.hand_orientations[self.another_hand_indices, :])[0].unsqueeze(-1)
        self.obs_buf[:, hand_another_pose_start+4:hand_another_pose_start+5] = get_euler_xyz(self.hand_orientations[self.another_hand_indices, :])[1].unsqueeze(-1)
        self.obs_buf[:, hand_another_pose_start+5:hand_another_pose_start+6] = get_euler_xyz(self.hand_orientations[self.another_hand_indices, :])[2].unsqueeze(-1)

        action_another_obs_start = hand_another_pose_start + 6
        self.obs_buf[:, action_another_obs_start:action_another_obs_start + 22] = self.actions[:, 26:]

        # object
        obj_obs_start = action_another_obs_start + 22  
        self.obs_buf[:, obj_obs_start:obj_obs_start + 7] = self.target_pose
        self.obs_buf[:, obj_obs_start + 7:obj_obs_start + 10] = self.target_linvel
        self.obs_buf[:, obj_obs_start + 10:obj_obs_start + 13] = self.vel_obs_scale * self.target_angvel
        
        pc = self.compute_point_cloud_state(10.0)

    def reset(self, env_ids):
        # randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        # generate random values
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), self.num_shadow_hand_dofs * 2 + 5), device=self.device)

        # reset object
        self.root_state_tensor[self.object1_1_indices[env_ids]] = self.object1_1_init_state[env_ids].clone()
        self.root_state_tensor[self.object1_2_indices[env_ids]] = self.object1_2_init_state[env_ids].clone()
        self.root_state_tensor[self.object1_3_indices[env_ids]] = self.object1_3_init_state[env_ids].clone()
        self.root_state_tensor[self.object1_4_indices[env_ids]] = self.object1_4_init_state[env_ids].clone()
        self.root_state_tensor[self.object2_1_indices[env_ids]] = self.object2_1_init_state[env_ids].clone()
        self.root_state_tensor[self.object2_2_indices[env_ids]] = self.object2_2_init_state[env_ids].clone()
        self.root_state_tensor[self.object2_3_indices[env_ids]] = self.object2_3_init_state[env_ids].clone()
        self.root_state_tensor[self.object2_4_indices[env_ids]] = self.object2_4_init_state[env_ids].clone()
        self.root_state_tensor[self.object3_1_indices[env_ids]] = self.object3_1_init_state[env_ids].clone()
        self.root_state_tensor[self.object3_2_indices[env_ids]] = self.object3_2_init_state[env_ids].clone()
        self.root_state_tensor[self.object3_3_indices[env_ids]] = self.object3_3_init_state[env_ids].clone()
        self.root_state_tensor[self.object3_4_indices[env_ids]] = self.object3_4_init_state[env_ids].clone()
        self.root_state_tensor[self.object4_1_indices[env_ids]] = self.object4_1_init_state[env_ids].clone()
        self.root_state_tensor[self.object4_2_indices[env_ids]] = self.object4_2_init_state[env_ids].clone()
        self.root_state_tensor[self.object4_3_indices[env_ids]] = self.object4_3_init_state[env_ids].clone()
        self.root_state_tensor[self.object4_4_indices[env_ids]] = self.object4_4_init_state[env_ids].clone()
        self.root_state_tensor[self.target_indices[env_ids]] = self.target_init_state[env_ids].clone()
        
        self.root_state_tensor[self.object1_1_indices[env_ids], 0:2] = self.object1_1_init_state[env_ids, 0:2] + self.reset_position_noise * rand_floats[:, 0:2]
        self.root_state_tensor[self.object1_2_indices[env_ids], 0:2] = self.object1_2_init_state[env_ids, 0:2] + self.reset_position_noise * rand_floats[:, 0:2]
        self.root_state_tensor[self.object1_3_indices[env_ids], 0:2] = self.object1_3_init_state[env_ids, 0:2] + self.reset_position_noise * rand_floats[:, 0:2]
        self.root_state_tensor[self.object1_4_indices[env_ids], 0:2] = self.object1_4_init_state[env_ids, 0:2] + self.reset_position_noise * rand_floats[:, 0:2]
        self.root_state_tensor[self.object2_1_indices[env_ids], 0:2] = self.object2_1_init_state[env_ids, 0:2] + self.reset_position_noise * rand_floats[:, 0:2]
        self.root_state_tensor[self.object2_2_indices[env_ids], 0:2] = self.object2_2_init_state[env_ids, 0:2] + self.reset_position_noise * rand_floats[:, 0:2]
        self.root_state_tensor[self.object2_3_indices[env_ids], 0:2] = self.object2_3_init_state[env_ids, 0:2] + self.reset_position_noise * rand_floats[:, 0:2]
        self.root_state_tensor[self.object2_4_indices[env_ids], 0:2] = self.object2_4_init_state[env_ids, 0:2] + self.reset_position_noise * rand_floats[:, 0:2]
        self.root_state_tensor[self.object3_1_indices[env_ids], 0:2] = self.object3_1_init_state[env_ids, 0:2] + self.reset_position_noise * rand_floats[:, 0:2]
        self.root_state_tensor[self.object3_2_indices[env_ids], 0:2] = self.object3_2_init_state[env_ids, 0:2] + self.reset_position_noise * rand_floats[:, 0:2]
        self.root_state_tensor[self.object3_3_indices[env_ids], 0:2] = self.object3_3_init_state[env_ids, 0:2] + self.reset_position_noise * rand_floats[:, 0:2]
        self.root_state_tensor[self.object3_4_indices[env_ids], 0:2] = self.object3_4_init_state[env_ids, 0:2] + self.reset_position_noise * rand_floats[:, 0:2]
        self.root_state_tensor[self.object4_1_indices[env_ids], 0:2] = self.object4_1_init_state[env_ids, 0:2] + self.reset_position_noise * rand_floats[:, 0:2]
        self.root_state_tensor[self.object4_2_indices[env_ids], 0:2] = self.object4_2_init_state[env_ids, 0:2] + self.reset_position_noise * rand_floats[:, 0:2]
        self.root_state_tensor[self.object4_3_indices[env_ids], 0:2] = self.object4_3_init_state[env_ids, 0:2] + self.reset_position_noise * rand_floats[:, 0:2]
        self.root_state_tensor[self.object4_4_indices[env_ids], 0:2] = self.object4_4_init_state[env_ids, 0:2] + self.reset_position_noise * rand_floats[:, 0:2]
        self.root_state_tensor[self.target_indices[env_ids], 0:2] = self.target_init_state[env_ids, 0:2] + self.reset_position_noise * rand_floats[:, 0:2]

        self.root_state_tensor[self.object1_1_indices[env_ids], self.up_axis_idx] = self.object1_1_init_state[env_ids, self.up_axis_idx] + self.reset_position_noise * rand_floats[:, self.up_axis_idx]
        self.root_state_tensor[self.object1_2_indices[env_ids], self.up_axis_idx] = self.object1_2_init_state[env_ids, self.up_axis_idx] + self.reset_position_noise * rand_floats[:, self.up_axis_idx]
        self.root_state_tensor[self.object1_3_indices[env_ids], self.up_axis_idx] = self.object1_3_init_state[env_ids, self.up_axis_idx] + self.reset_position_noise * rand_floats[:, self.up_axis_idx]
        self.root_state_tensor[self.object1_4_indices[env_ids], self.up_axis_idx] = self.object1_4_init_state[env_ids, self.up_axis_idx] + self.reset_position_noise * rand_floats[:, self.up_axis_idx]
        self.root_state_tensor[self.object2_1_indices[env_ids], self.up_axis_idx] = self.object2_1_init_state[env_ids, self.up_axis_idx] + self.reset_position_noise * rand_floats[:, self.up_axis_idx]
        self.root_state_tensor[self.object2_2_indices[env_ids], self.up_axis_idx] = self.object2_2_init_state[env_ids, self.up_axis_idx] + self.reset_position_noise * rand_floats[:, self.up_axis_idx]
        self.root_state_tensor[self.object2_3_indices[env_ids], self.up_axis_idx] = self.object2_3_init_state[env_ids, self.up_axis_idx] + self.reset_position_noise * rand_floats[:, self.up_axis_idx]
        self.root_state_tensor[self.object2_4_indices[env_ids], self.up_axis_idx] = self.object2_4_init_state[env_ids, self.up_axis_idx] + self.reset_position_noise * rand_floats[:, self.up_axis_idx]
        self.root_state_tensor[self.object3_1_indices[env_ids], self.up_axis_idx] = self.object3_1_init_state[env_ids, self.up_axis_idx] + self.reset_position_noise * rand_floats[:, self.up_axis_idx]
        self.root_state_tensor[self.object3_2_indices[env_ids], self.up_axis_idx] = self.object3_2_init_state[env_ids, self.up_axis_idx] + self.reset_position_noise * rand_floats[:, self.up_axis_idx]
        self.root_state_tensor[self.object3_3_indices[env_ids], self.up_axis_idx] = self.object3_3_init_state[env_ids, self.up_axis_idx] + self.reset_position_noise * rand_floats[:, self.up_axis_idx]
        self.root_state_tensor[self.object3_4_indices[env_ids], self.up_axis_idx] = self.object3_4_init_state[env_ids, self.up_axis_idx] + self.reset_position_noise * rand_floats[:, self.up_axis_idx]
        self.root_state_tensor[self.object4_1_indices[env_ids], self.up_axis_idx] = self.object4_1_init_state[env_ids, self.up_axis_idx] + self.reset_position_noise * rand_floats[:, self.up_axis_idx]
        self.root_state_tensor[self.object4_2_indices[env_ids], self.up_axis_idx] = self.object4_2_init_state[env_ids, self.up_axis_idx] + self.reset_position_noise * rand_floats[:, self.up_axis_idx]
        self.root_state_tensor[self.object4_3_indices[env_ids], self.up_axis_idx] = self.object4_3_init_state[env_ids, self.up_axis_idx] + self.reset_position_noise * rand_floats[:, self.up_axis_idx]
        self.root_state_tensor[self.object4_4_indices[env_ids], self.up_axis_idx] = self.object4_4_init_state[env_ids, self.up_axis_idx] + self.reset_position_noise * rand_floats[:, self.up_axis_idx]
        self.root_state_tensor[self.target_indices[env_ids], self.up_axis_idx] = self.target_init_state[env_ids, self.up_axis_idx] + self.reset_position_noise * rand_floats[:, self.up_axis_idx]

        self.root_state_tensor[self.object1_1_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.object1_1_indices[env_ids], 7:13])
        self.root_state_tensor[self.object1_2_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.object1_2_indices[env_ids], 7:13])
        self.root_state_tensor[self.object1_3_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.object1_3_indices[env_ids], 7:13])
        self.root_state_tensor[self.object1_4_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.object1_4_indices[env_ids], 7:13])
        self.root_state_tensor[self.object2_1_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.object2_1_indices[env_ids], 7:13])
        self.root_state_tensor[self.object2_2_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.object2_2_indices[env_ids], 7:13])
        self.root_state_tensor[self.object2_3_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.object2_3_indices[env_ids], 7:13])
        self.root_state_tensor[self.object2_4_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.object2_4_indices[env_ids], 7:13])
        self.root_state_tensor[self.object3_1_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.object3_1_indices[env_ids], 7:13])
        self.root_state_tensor[self.object3_2_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.object3_2_indices[env_ids], 7:13])
        self.root_state_tensor[self.object3_3_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.object3_3_indices[env_ids], 7:13])
        self.root_state_tensor[self.object3_4_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.object3_4_indices[env_ids], 7:13])
        self.root_state_tensor[self.object4_1_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.object4_1_indices[env_ids], 7:13])
        self.root_state_tensor[self.object4_2_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.object4_2_indices[env_ids], 7:13])
        self.root_state_tensor[self.object4_3_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.object4_3_indices[env_ids], 7:13])
        self.root_state_tensor[self.object4_4_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.object4_4_indices[env_ids], 7:13])
        self.root_state_tensor[self.target_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.target_indices[env_ids], 7:13])

        object1_1_indices = torch.unique(torch.cat([self.object1_1_indices[env_ids]]))
        object1_2_indices = torch.unique(torch.cat([self.object1_2_indices[env_ids]]))
        object1_3_indices = torch.unique(torch.cat([self.object1_3_indices[env_ids]]))
        object1_4_indices = torch.unique(torch.cat([self.object1_4_indices[env_ids]]))
        object2_1_indices = torch.unique(torch.cat([self.object2_1_indices[env_ids]]))
        object2_2_indices = torch.unique(torch.cat([self.object2_2_indices[env_ids]]))
        object2_3_indices = torch.unique(torch.cat([self.object2_3_indices[env_ids]]))
        object2_4_indices = torch.unique(torch.cat([self.object2_4_indices[env_ids]]))
        object3_1_indices = torch.unique(torch.cat([self.object3_1_indices[env_ids]]))
        object3_2_indices = torch.unique(torch.cat([self.object3_2_indices[env_ids]]))
        object3_3_indices = torch.unique(torch.cat([self.object3_3_indices[env_ids]]))
        object3_4_indices = torch.unique(torch.cat([self.object3_4_indices[env_ids]]))
        object4_1_indices = torch.unique(torch.cat([self.object4_1_indices[env_ids]]))
        object4_2_indices = torch.unique(torch.cat([self.object4_2_indices[env_ids]]))
        object4_3_indices = torch.unique(torch.cat([self.object4_3_indices[env_ids]]))
        object4_4_indices = torch.unique(torch.cat([self.object4_4_indices[env_ids]]))
        target_indices = torch.unique(torch.cat([self.target_indices[env_ids]]))

        # reset shadow hand and allegro hand
        delta_max_shadow = self.shadow_hand_dof_upper_limits - self.shadow_hand_dof_default_pos
        delta_min_shadow = self.shadow_hand_dof_lower_limits - self.shadow_hand_dof_default_pos
        rand_delta_shadow = delta_min_shadow + (delta_max_shadow - delta_min_shadow) * rand_floats[:, 5:5+self.num_shadow_hand_dofs]

        delta_max_allegro = self.allegro_hand_dof_upper_limits - self.allegro_hand_dof_default_pos
        delta_min_allegro = self.allegro_hand_dof_lower_limits - self.allegro_hand_dof_default_pos
        rand_delta_allegro = delta_min_allegro+ (delta_max_allegro - delta_min_allegro) * rand_floats[:, 5:5+self.num_allegro_hand_dofs]

        shadow_hand_pos = self.shadow_hand_default_dof_pos + self.reset_dof_pos_noise * rand_delta_shadow
        allegro_hand_pos = self.allegro_hand_default_dof_pos + self.reset_dof_pos_noise * rand_delta_allegro

        self.shadow_hand_dof_pos[env_ids, :] = shadow_hand_pos
        self.allegro_hand_dof_pos[env_ids, :] = allegro_hand_pos

        self.shadow_hand_dof_vel[env_ids, :] = self.shadow_hand_dof_default_vel + \
            self.reset_dof_vel_noise * rand_floats[:, 5+self.num_shadow_hand_dofs:5+self.num_shadow_hand_dofs*2]   

        self.allegro_hand_dof_vel[env_ids, :] = self.allegro_hand_dof_default_vel + \
            self.reset_dof_vel_noise * rand_floats[:, 5+self.num_allegro_hand_dofs:5+self.num_allegro_hand_dofs*2]

        self.prev_targets[env_ids, :self.num_shadow_hand_dofs] = shadow_hand_pos
        self.cur_targets[env_ids, :self.num_shadow_hand_dofs] = shadow_hand_pos

        self.prev_targets[env_ids, self.num_allegro_hand_dofs:self.num_allegro_hand_dofs*2] = allegro_hand_pos
        self.cur_targets[env_ids, self.num_allegro_hand_dofs:self.num_allegro_hand_dofs*2] = allegro_hand_pos

        # self.prev_targets[env_ids, 48:48 + 2] = to_torch([0, 0], device=self.device)
        # self.cur_targets[env_ids, self.num_shadow_hand_dofs*2:self.num_shadow_hand_dofs*2 + 2] = to_torch([0, 0], device=self.device)

        hand_indices = self.hand_indices[env_ids].to(torch.int32)
        another_hand_indices = self.another_hand_indices[env_ids].to(torch.int32)

        all_hand_indices = torch.unique(torch.cat([hand_indices,
                                                 another_hand_indices, 
                                                 ]).to(torch.int32))

        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.prev_targets),
                                                        gymtorch.unwrap_tensor(all_hand_indices), len(all_hand_indices))  

        self.hand_positions[all_hand_indices.to(torch.long), :] = self.saved_root_tensor[all_hand_indices.to(torch.long), 0:3]
        self.hand_orientations[all_hand_indices.to(torch.long), :] = self.saved_root_tensor[all_hand_indices.to(torch.long), 3:7]
        self.hand_angvels[all_hand_indices.to(torch.long), :] = self.saved_root_tensor[all_hand_indices.to(torch.long), 7:10]
        self.hand_linvels[all_hand_indices.to(torch.long), :] = self.saved_root_tensor[all_hand_indices.to(torch.long), 10:13]

        all_indices = torch.unique(torch.cat([all_hand_indices,
                                              object1_1_indices,
                                              object1_2_indices, 
                                              object1_3_indices, 
                                              object1_4_indices, 
                                              object2_1_indices,
                                              object2_2_indices, 
                                              object2_3_indices, 
                                              object2_4_indices, 
                                              object3_1_indices,
                                              object3_2_indices, 
                                              object3_3_indices, 
                                              object3_4_indices, 
                                              object4_1_indices,
                                              object4_2_indices, 
                                              object4_3_indices, 
                                              object4_4_indices, 
                                              target_indices,
                                              self.table_indices[env_ids]]).to(torch.int32))


        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(all_hand_indices), len(all_hand_indices))
                                              
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_state_tensor),
                                                     gymtorch.unwrap_tensor(all_indices), len(all_indices))
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.successes[env_ids] = 0


    def pre_physics_step(self, actions):
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        
        if len(env_ids) > 0:
            self.reset(env_ids)

        self.actions = actions.clone().to(self.device)
        if self.use_relative_control:
            targets = self.prev_targets[:, self.actuated_dof_indices] + self.shadow_hand_dof_speed_scale * self.dt * self.actions
            self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(targets,
                                                                          self.shadow_hand_dof_lower_limits[self.actuated_dof_indices], self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])
        else:
            self.cur_targets[:, self.shadow_hand_actuated_dof_indices] = scale(self.actions[:, 6:26],
                                                                   self.shadow_hand_dof_lower_limits[self.shadow_hand_actuated_dof_indices], self.shadow_hand_dof_upper_limits[self.shadow_hand_actuated_dof_indices])
            self.cur_targets[:, self.shadow_hand_actuated_dof_indices] = self.act_moving_average * self.cur_targets[:,
                                                                                                        self.shadow_hand_actuated_dof_indices] + (1.0 - self.act_moving_average) * self.prev_targets[:, self.shadow_hand_actuated_dof_indices]
            self.cur_targets[:, self.shadow_hand_actuated_dof_indices] = tensor_clamp(self.cur_targets[:, self.shadow_hand_actuated_dof_indices],
                                                                          self.shadow_hand_dof_lower_limits[self.shadow_hand_actuated_dof_indices], self.shadow_hand_dof_upper_limits[self.shadow_hand_actuated_dof_indices])

            self.cur_targets[:, self.allegro_hand_actuated_dof_indices + 24] = scale(self.actions[:, 32:48],
                                                                   self.allegro_hand_dof_lower_limits[self.allegro_hand_actuated_dof_indices], self.allegro_hand_dof_upper_limits[self.allegro_hand_actuated_dof_indices])
            self.cur_targets[:, self.allegro_hand_actuated_dof_indices + 24] = self.act_moving_average * self.cur_targets[:,
                                                                                                        self.allegro_hand_actuated_dof_indices + 24] + (1.0 - self.act_moving_average) * self.prev_targets[:, self.allegro_hand_actuated_dof_indices]
            self.cur_targets[:, self.allegro_hand_actuated_dof_indices + 24] = tensor_clamp(self.cur_targets[:, self.allegro_hand_actuated_dof_indices + 24],
                                                                          self.allegro_hand_dof_lower_limits[self.allegro_hand_actuated_dof_indices], self.allegro_hand_dof_upper_limits[self.allegro_hand_actuated_dof_indices])

            self.apply_forces[:, 1, :] = actions[:, 0:3] * self.dt * self.transition_scale * 10000000
            self.apply_forces[:, 26, :] = actions[:, 26:29] * self.dt * self.transition_scale * 10000000
            
            self.apply_torque[:, 1, :] = self.actions[:, 3:6] * self.dt * self.orientation_scale * 100000
            self.apply_torque[:, 26, :] = self.actions[:, 29:32] * self.dt * self.orientation_scale * 100000
            self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.apply_forces), gymtorch.unwrap_tensor(self.apply_torque), gymapi.ENV_SPACE)

        self.prev_targets[:, self.shadow_hand_actuated_dof_indices] = self.cur_targets[:, self.shadow_hand_actuated_dof_indices]
        self.prev_targets[:, self.allegro_hand_actuated_dof_indices + 24] = self.cur_targets[:, self.allegro_hand_actuated_dof_indices + 24]
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.cur_targets))


    def post_physics_step(self):
        self.progress_buf += 1
        self.randomize_buf += 1

        self.compute_observations()
        self.compute_reward(self.actions)
        self.compute_cost()

    def add_debug_lines(self, env, pos, rot):
        posx = (pos + quat_apply(rot, to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
        posy = (pos + quat_apply(rot, to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
        posz = (pos + quat_apply(rot, to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

        p0 = pos.cpu().numpy()
        self.gym.add_lines(self.viewer, env, 1, [p0[0], p0[1], p0[2], posx[0], posx[1], posx[2]], [0.85, 0.1, 0.1])
        self.gym.add_lines(self.viewer, env, 1, [p0[0], p0[1], p0[2], posy[0], posy[1], posy[2]], [0.1, 0.85, 0.1])
        self.gym.add_lines(self.viewer, env, 1, [p0[0], p0[1], p0[2], posz[0], posz[1], posz[2]], [0.1, 0.1, 0.85])
    def depth_image_to_point_cloud_GPU(self, camera_tensor, rgb, camera_view_matrix_inv, camera_proj_matrix, u, v, width, height, depth_bar):
        # time1 = time.time()
        depth_buffer = camera_tensor.to(self.device)

        # Get the camera view matrix and invert it to transform points from camera to world space
        vinv = camera_view_matrix_inv
        # Get the camera projection matrix and get the necessary scaling
        # coefficients for deprojection
        proj = camera_proj_matrix
        fu = 2/proj[0, 0]
        fv = 2/proj[1, 1]
        centerU = width/2
        centerV = height/2
        Z = depth_buffer
        X = -(u-centerU)/width * Z * fu
        Y = (v-centerV)/height * Z * fv
        Z = Z.view(-1)
        valid = Z > -depth_bar
        X = X.view(-1)
        Y = Y.view(-1)
        r = rgb[:, :, 0].view(-1)[valid].reshape(-1, 1)
        g = rgb[:, :, 1].view(-1)[valid].reshape(-1, 1)
        b = rgb[:, :, 2].view(-1)[valid].reshape(-1, 1)
        position = torch.vstack((X, Y, Z, torch.ones(len(X), device=self.device)))[:, valid]
        position = position.permute(1, 0)
        position = position@vinv
        points = torch.cat((position[:, :3], r, g, b), 1)

        return points

    def sample_points(self, points, sample_num=1000, sample_method='random'):
        eff_points = points[points[:, 2]>0.04]
        sampled_points_id = pointnet2_utils.furthest_point_sample(eff_points.reshape(1, *eff_points.shape), sample_num)
        sampled_points = eff_points.index_select(0, sampled_points_id[0].long())
        return sampled_points

    def compute_point_cloud_state(self, depth_bar):
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)
        num_envs=self.num_envs
        pointCloudDownsampleNum=2048*5
        point_clouds = torch.zeros((num_envs, pointCloudDownsampleNum, 6), device=self.device)
        
        for i in range(num_envs):

            points1 = self.depth_image_to_point_cloud_GPU(self.camera_tensor_list_depth[i][0], self.camera_tensor_list_color[i][0][:, :, :3], self.camera_view_matrix_inv_list[i][0], self.camera_proj_matrix_list[i][0], self.camera_u2, self.camera_v2, self.camera_props.width, self.camera_props.height, depth_bar)
            points2 = self.depth_image_to_point_cloud_GPU(self.camera_tensor_list_depth[i][1], self.camera_tensor_list_color[i][1][:, :, :3], self.camera_view_matrix_inv_list[i][1], self.camera_proj_matrix_list[i][1], self.camera_u2, self.camera_v2, self.camera_props.width, self.camera_props.height, depth_bar)
            points3 = self.depth_image_to_point_cloud_GPU(self.camera_tensor_list_depth[i][2], self.camera_tensor_list_color[i][2][:, :, :3], self.camera_view_matrix_inv_list[i][2], self.camera_proj_matrix_list[i][2], self.camera_u2, self.camera_v2, self.camera_props.width, self.camera_props.height, depth_bar)
            points = torch.cat((points1,), )#points2, points3, points2, points3

            selected_points = self.sample_points(points, sample_num=pointCloudDownsampleNum, sample_method='furthest')
            point_clouds[i]=selected_points


        self.gym.end_access_image_tensors(self.sim)
        point_clouds_xyz = point_clouds[:, :, :3]
        point_clouds_rgb = point_clouds[:, :, 3:]
        point_clouds_xyz -= self.env_origin.view(num_envs, 1, 3)
        print(point_clouds.shape)
        self.camera_time+=1
        
        torch.save(point_clouds, "temp/jenda_pc{}.pth".format(self.camera_time))
        torch.save(self.camera_tensor_list_depth, "temp/depth{}.pth".format(self.camera_time))
        torch.save(self.camera_tensor_list_color, "temp/color{}.pth".format(self.camera_time))

        return point_clouds

#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_hand_reward(
    rew_buf, reset_buf, progress_buf, successes, consecutive_successes,
    max_episode_length: float,
    target_pos, target_rot,
    object1_1_pos, object1_2_pos, object1_3_pos, object1_4_pos, 
    object2_1_pos, object2_2_pos, object2_3_pos, object2_4_pos, 
    object3_1_pos, object3_2_pos, object3_3_pos, object3_4_pos, 
    object4_1_pos, object4_2_pos, object4_3_pos, object4_4_pos, 
    stable1_1_pos, stable1_2_pos, stable1_3_pos, stable1_4_pos, 
    stable2_1_pos, stable2_2_pos, stable2_3_pos, stable2_4_pos, 
    stable3_1_pos, stable3_2_pos, stable3_3_pos, stable3_4_pos, 
    stable4_1_pos, stable4_2_pos, stable4_3_pos, stable4_4_pos, 
    right_end_pos, left_end_pos,
    left_hand_pos, right_hand_pos, 
    right_hand_ff_pos, right_hand_mf_pos, right_hand_rf_pos, right_hand_lf_pos, right_hand_th_pos,
    left_hand_ff_pos, left_hand_mf_pos, left_hand_lf_pos, left_hand_th_pos,
    dist_reward_scale: float, rot_reward_scale: float, rot_eps: float,
    actions, action_penalty_scale: float,
    success_tolerance: float, fall_dist: float,
    fall_penalty: float, max_consecutive_successes: int, av_factor: float
):

    action_penalty = torch.sum(actions ** 2, dim=-1)

    # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
    # reward = torch.exp(-0.05*(up_rew * dist_reward_scale)) + torch.exp(-0.05*(right_hand_dist_rew * dist_reward_scale)) + torch.exp(-0.05*(left_hand_dist_rew * dist_reward_scale))

    move_rew =  (target_pos[:, 1]+0.6)*30
    
    # move_rew = torch.where(move_rew>1.0, torch.ones_like(move_rew)*1.0, move_rew)
    
    delta_1_1 = torch.norm(object1_1_pos-stable1_1_pos, dim=1)
    delta_1_2 = torch.norm(object1_2_pos-stable1_2_pos, dim=1)
    delta_1_3 = torch.norm(object1_3_pos-stable1_3_pos, dim=1)
    delta_1_4 = torch.norm(object1_4_pos-stable1_4_pos, dim=1)
    delta_2_1 = torch.norm(object2_1_pos-stable2_1_pos, dim=1)
    delta_2_2 = torch.norm(object2_2_pos-stable2_2_pos, dim=1)
    delta_2_3 = torch.norm(object2_3_pos-stable2_3_pos, dim=1)
    delta_2_4 = torch.norm(object2_4_pos-stable2_4_pos, dim=1)
    delta_3_1 = torch.norm(object3_1_pos-stable3_1_pos, dim=1)
    delta_3_2 = torch.norm(object3_2_pos-stable3_2_pos, dim=1)
    delta_3_3 = torch.norm(object3_3_pos-stable3_3_pos, dim=1)
    delta_3_4 = torch.norm(object3_4_pos-stable3_4_pos, dim=1)
    delta_4_1 = torch.norm(object4_1_pos-stable4_1_pos, dim=1)
    delta_4_2 = torch.norm(object4_2_pos-stable4_2_pos, dim=1)
    delta_4_3 = torch.norm(object4_3_pos-stable4_3_pos, dim=1)
    delta_4_4 = torch.norm(object4_4_pos-stable4_4_pos, dim=1)

    dz_1_1 = abs(object1_1_pos[:, 2] - stable1_1_pos[2])
    dz_1_2 = abs(object1_2_pos[:, 2] - stable1_2_pos[2])
    dz_1_3 = abs(object1_3_pos[:, 2] - stable1_3_pos[2])
    dz_1_4 = abs(object1_4_pos[:, 2] - stable1_4_pos[2])
    dz_2_1 = abs(object2_1_pos[:, 2] - stable2_1_pos[2])
    dz_2_2 = abs(object2_2_pos[:, 2] - stable2_2_pos[2])
    dz_2_3 = abs(object2_3_pos[:, 2] - stable2_3_pos[2])
    dz_2_4 = abs(object2_4_pos[:, 2] - stable2_4_pos[2])
    dz_3_1 = abs(object3_1_pos[:, 2] - stable3_1_pos[2])
    dz_3_2 = abs(object3_2_pos[:, 2] - stable3_2_pos[2])
    dz_3_3 = abs(object3_3_pos[:, 2] - stable3_3_pos[2])
    dz_3_4 = abs(object3_4_pos[:, 2] - stable3_4_pos[2])
    dz_4_1 = abs(object4_1_pos[:, 2] - stable4_1_pos[2])
    dz_4_2 = abs(object4_2_pos[:, 2] - stable4_2_pos[2])
    dz_4_3 = abs(object4_3_pos[:, 2] - stable4_3_pos[2])
    dz_4_4 = abs(object4_4_pos[:, 2] - stable4_4_pos[2])

    move_others = (delta_1_1 + delta_1_2 + delta_1_3 + delta_1_4 + \
                    delta_2_1 + delta_2_2 + delta_2_3 + delta_2_4 + \
                    delta_3_1 + delta_3_2 + delta_3_3 + delta_3_4 + \
                    delta_4_1 + delta_4_2 + delta_4_3 + delta_4_4)
    exp_move_penalty = torch.exp(move_others)-1
    move_penalty = -exp_move_penalty*4
    
    left_hand_finger_dist = (torch.norm(right_end_pos - left_hand_pos, p=2, dim=-1))
    right_hand_finger_dist = (torch.norm(left_end_pos - right_hand_mf_pos, p=2, dim=-1))
    
    dist_reward = -left_hand_finger_dist/0.2 - right_hand_finger_dist/0.5
    
    loss =  (dz_1_1>0.01)*(dz_1_2>0.01)*(dz_1_3>0.01)*(dz_1_4>0.01)*\
            (dz_2_1>0.01)*(dz_2_2>0.01)*(dz_2_3>0.01)*(dz_2_4>0.01)*\
            (dz_3_1>0.01)*(dz_3_2>0.01)*(dz_3_3>0.01)*(dz_3_4>0.01)*\
            (dz_4_1>0.01)*(dz_4_2>0.01)*(dz_4_3>0.01)*(dz_4_4>0.01)
            
    reward =  move_rew + move_penalty + dist_reward + 0.5

    print("move_reward_mean:", move_rew.mean().item())
    print("move_penalty_mean:", move_penalty.mean().item())
    print("dist_reward_mean:", dist_reward.mean().item())
    print("total_reward_mean:", reward.mean().item())
    print()
    print("move_reward_0:", move_rew[0].item())
    print("move_penalty_0:", move_penalty[0].item())
    print("dist_reward_0:", dist_reward[0].item())
    print("total_reward_0:", reward[0].item())
    print("if_loss_0:", loss[0].item())
    print()
    resets = torch.where(loss, torch.ones_like(reset_buf), reset_buf)
    resets = torch.where(move_penalty <= -0.4, torch.ones_like(reset_buf), resets)

    # Find out which envs hit the goal and update successes count
    resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets)
    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())

    cons_successes = torch.where(num_resets > 0, av_factor*finished_cons_successes/num_resets + (1.0 - av_factor)*consecutive_successes, consecutive_successes)

    return reward, resets,  progress_buf, successes, cons_successes


@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(quat_from_angle_axis(rand0 * np.pi, x_unit_tensor),
                    quat_from_angle_axis(rand1 * np.pi, y_unit_tensor))


@torch.jit.script
def randomize_rotation_pen(rand0, rand1, max_angle, x_unit_tensor, y_unit_tensor, z_unit_tensor):
    rot = quat_mul(quat_from_angle_axis(0.5 * np.pi + rand0 * max_angle, x_unit_tensor),
                   quat_from_angle_axis(rand0 * np.pi, z_unit_tensor))
    return rot

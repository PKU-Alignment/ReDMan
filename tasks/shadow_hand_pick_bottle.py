# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

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
class ShadowHandPickBottle(BaseTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless, agent_index=[[[0, 1, 2, 3, 4, 5]], [[0, 1, 2, 3, 4, 5]]], is_multi_agent=False, algo='ppol'):
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

        self.object_type = self.cfg["env"]["objectType"]
        # assert self.object_type in ["block", "egg", "pen"]

        self.ignore_z = (self.object_type == "pen")

        self.asset_files_dict = {
            "block": "urdf/objects/cube_multicolor.urdf",
            "egg": "mjcf/open_ai_assets/hand/egg.xml",
            "pen": "mjcf/open_ai_assets/hand/pen.xml",
            "bottle_cap": "mjcf/bottle_cap/mobility.urdf",
            "table": "urdf/table/mobility_pick_bottle.urdf"
        }

        if "asset" in self.cfg["env"]:
            self.asset_files_dict["block"] = self.cfg["env"]["asset"].get("assetFileNameBlock", self.asset_files_dict["block"])
            self.asset_files_dict["egg"] = self.cfg["env"]["asset"].get("assetFileNameEgg", self.asset_files_dict["egg"])
            self.asset_files_dict["pen"] = self.cfg["env"]["asset"].get("assetFileNamePen", self.asset_files_dict["pen"])

        # can be "openai", "full_no_vel", "full", "full_state"
        self.obs_type = self.cfg["env"]["observationType"]

        if not (self.obs_type in ["openai", "full_no_vel", "full", "full_state", 'pick_bottle']):
            raise Exception(
                "Unknown type of observations!\nobservationType should be one of: [openai, full_no_vel, full, full_state]")

        print("Obs type:", self.obs_type)

        self.num_obs_dict = {
            "openai": 42,
            "full_no_vel": 77,
            "full": 157,
            "full_state": 422 - 11 + 6 + 3, 
            "pick_bottle": 430
        }
        self.num_hand_obs = 72 + 95 + 26 + 6
        self.up_axis = 'z'

        self.fingertips = ["robot0:ffdistal", "robot0:mfdistal", "robot0:rfdistal", "robot0:lfdistal", "robot0:thdistal"]
        self.a_fingertips = ["robot1:ffdistal", "robot1:mfdistal", "robot1:rfdistal", "robot1:lfdistal", "robot1:thdistal"]
        self.hand_center = ["robot1:palm"]
        self.num_fingertips = len(self.fingertips) * 2
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
            self.cfg["env"]["numActions"] = 52

        self.cfg["device_type"] = device_type
        self.cfg["device_id"] = device_id
        self.cfg["headless"] = headless

        super().__init__(cfg=self.cfg)

        if self.viewer != None:
            cam_pos = gymapi.Vec3(1.0, 0.0, 1.0)
            cam_target = gymapi.Vec3(0.0, 0.0, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        if self.obs_type == "full_state" or self.obs_type == "pick_bottle" or self.asymmetric_obs:
            sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
            self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, self.num_fingertips * 6)

            dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
            self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs, self.num_shadow_hand_dofs * 2)
            self.dof_force_tensor = self.dof_force_tensor[:, :48]

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.shadow_hand_default_dof_pos = torch.zeros(self.num_shadow_hand_dofs, dtype=torch.float, device=self.device)
        self.shadow_hand_default_dof_pos = to_torch([0.0, 0.0, -0,  -0,  -0,  -0, -0, -0,
                                            -0,  -0, -0,  -0,  -0,  -0, -0, -0,
                                            -0,  -0, -0,  -1.04,  1.2,  0., 0, -1.57], dtype=torch.float, device=self.device)

        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.shadow_hand_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_shadow_hand_dofs]
        self.shadow_hand_dof_pos = self.shadow_hand_dof_state[..., 0]
        self.shadow_hand_dof_vel = self.shadow_hand_dof_state[..., 1]

        self.shadow_hand_another_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, self.num_shadow_hand_dofs:self.num_shadow_hand_dofs*2]
        self.shadow_hand_another_dof_pos = self.shadow_hand_another_dof_state[..., 0]
        self.shadow_hand_another_dof_vel = self.shadow_hand_another_dof_state[..., 1]

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
        shadow_hand_asset_file = "mjcf/open_ai_assets/hand/shadow_hand.xml"
        shadow_hand_another_asset_file = "mjcf/open_ai_assets/hand/shadow_hand1.xml"

        if "asset" in self.cfg["env"]:
            asset_root = self.cfg["env"]["asset"].get("assetRoot", asset_root)
            shadow_hand_asset_file = self.cfg["env"]["asset"].get("assetFileName", shadow_hand_asset_file)

        object_asset_file = self.asset_files_dict[self.object_type]

        # load shadow hand_ asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = False
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.angular_damping = 100
        asset_options.linear_damping = 100

        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        shadow_hand_asset = self.gym.load_asset(self.sim, asset_root, shadow_hand_asset_file, asset_options)
        shadow_hand_another_asset = self.gym.load_asset(self.sim, asset_root, shadow_hand_another_asset_file, asset_options)

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

        # tendon set up
        limit_stiffness = 30
        t_damping = 0.1
        relevant_tendons = ["robot0:T_FFJ1c", "robot0:T_MFJ1c", "robot0:T_RFJ1c", "robot0:T_LFJ1c"]
        a_relevant_tendons = ["robot1:T_FFJ1c", "robot1:T_MFJ1c", "robot1:T_RFJ1c", "robot1:T_LFJ1c"]
        tendon_props = self.gym.get_asset_tendon_properties(shadow_hand_asset)
        a_tendon_props = self.gym.get_asset_tendon_properties(shadow_hand_another_asset)

        for i in range(self.num_shadow_hand_tendons):
            for rt in relevant_tendons:
                if self.gym.get_asset_tendon_name(shadow_hand_asset, i) == rt:
                    tendon_props[i].limit_stiffness = limit_stiffness
                    tendon_props[i].damping = t_damping
            for rt in a_relevant_tendons:
                if self.gym.get_asset_tendon_name(shadow_hand_another_asset, i) == rt:
                    a_tendon_props[i].limit_stiffness = limit_stiffness
                    a_tendon_props[i].damping = t_damping
        self.gym.set_asset_tendon_properties(shadow_hand_asset, tendon_props)
        self.gym.set_asset_tendon_properties(shadow_hand_another_asset, a_tendon_props)
        
        actuated_dof_names = [self.gym.get_asset_actuator_joint_name(shadow_hand_asset, i) for i in range(self.num_shadow_hand_actuators)]
        self.actuated_dof_indices = [self.gym.find_asset_dof_index(shadow_hand_asset, name) for name in actuated_dof_names]

        # set shadow_hand dof properties
        shadow_hand_dof_props = self.gym.get_asset_dof_properties(shadow_hand_asset)
        shadow_hand_another_dof_props = self.gym.get_asset_dof_properties(shadow_hand_another_asset)

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

        self.actuated_dof_indices = to_torch(self.actuated_dof_indices, dtype=torch.long, device=self.device)
        self.shadow_hand_dof_lower_limits = to_torch(self.shadow_hand_dof_lower_limits, device=self.device)
        self.shadow_hand_dof_upper_limits = to_torch(self.shadow_hand_dof_upper_limits, device=self.device)
        self.shadow_hand_dof_default_pos = to_torch(self.shadow_hand_dof_default_pos, device=self.device)
        self.shadow_hand_dof_default_vel = to_torch(self.shadow_hand_dof_default_vel, device=self.device)

        # load manipulated object and goal assets
        moving_object_asset_options = gymapi.AssetOptions()
        moving_object_asset_options.density = 50
        fixed_object_asset_options = gymapi.AssetOptions()
        fixed_object_asset_options.density = 50
        fixed_object_asset_options.fix_base_link = True
        moving_object_asset = self.gym.load_asset(self.sim, asset_root, object_asset_file, moving_object_asset_options)
        fixed_object_asset = self.gym.load_asset(self.sim, asset_root, object_asset_file, fixed_object_asset_options)

        # set object dof properties
        self.num_object_dofs = self.gym.get_asset_dof_count(moving_object_asset)
        object_dof_props = self.gym.get_asset_dof_properties(moving_object_asset)

        self.object_dof_lower_limits = []
        self.object_dof_upper_limits = []
        self.object_dof_default_pos = []
        self.object_dof_default_vel = []

        for i in range(self.gym.get_asset_dof_count(moving_object_asset)):
            self.object_dof_lower_limits.append(object_dof_props['lower'][i])
            self.object_dof_upper_limits.append(object_dof_props['upper'][i])
            self.object_dof_default_pos.append(0.0)
            self.object_dof_default_vel.append(0.0)

        self.object_dof_lower_limits = to_torch(self.object_dof_lower_limits, device=self.device)
        self.object_dof_upper_limits = to_torch(self.object_dof_upper_limits, device=self.device)
        self.object_dof_default_pos = to_torch(self.object_dof_default_pos, device=self.device)
        self.object_dof_default_vel = to_torch(self.object_dof_default_vel, device=self.device)
        
        
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
        
        
        self.num_object_bodies = self.gym.get_asset_rigid_body_count(moving_object_asset)
        self.num_table_bodies = self.gym.get_asset_rigid_body_count(table_asset)
        self.num_object_shapes = self.gym.get_asset_rigid_shape_count(moving_object_asset)
        self.num_table_shapes = self.gym.get_asset_rigid_shape_count(table_asset)

        shadow_hand_start_pose = gymapi.Transform()
        shadow_hand_start_pose.p = gymapi.Vec3(-0.37, -0.74, 0.68)
        shadow_hand_start_pose.r = gymapi.Quat(0,0,1,1)*gymapi.Quat(0,1,0,0)

        shadow_another_hand_start_pose = gymapi.Transform()
        shadow_another_hand_start_pose.p = gymapi.Vec3(-0.37, -0.54, 0.68)
        shadow_another_hand_start_pose.r = gymapi.Quat(0,0,1,1)*gymapi.Quat(0,1,0,0)

        object_start_pose1 = gymapi.Transform()
        object_start_pose1.p = gymapi.Vec3(0, -0.80, 0.47278)
        object_start_pose1.r = gymapi.Quat(0,0,0,1)
        delta_start_pose = gymapi.Vec3(0, 0.1, 0)
        
        object_start_pose2 = copy.deepcopy(object_start_pose1)
        object_start_pose2.p += delta_start_pose
        object_start_pose3 = copy.deepcopy(object_start_pose2)
        object_start_pose3.p += delta_start_pose
        object_start_pose4 = copy.deepcopy(object_start_pose3)
        object_start_pose4.p += delta_start_pose
        object_start_pose5 = copy.deepcopy(object_start_pose4)
        object_start_pose5.p += delta_start_pose

        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(0.0, -0.6, 0.2)
        table_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # compute aggregate size
        max_agg_bodies = self.num_shadow_hand_bodies * 2 + 5 * self.num_object_bodies + self.num_table_bodies
        max_agg_shapes = self.num_shadow_hand_shapes * 2 + 5 * self.num_object_shapes + self.num_table_shapes

        self.shadow_hands = []
        self.envs = []
        self.object1_init_state = []
        self.object2_init_state = []
        self.object3_init_state = []
        self.object4_init_state = []
        self.object5_init_state = []
        self.hand_start_states = []
        self.hand_indices = []
        self.another_hand_indices = []
        self.fingertip_indices = []
        self.object1_indices = []
        self.object2_indices = []
        self.object3_indices = []
        self.object4_indices = []
        self.object5_indices = []
        self.table_indices = []
        self.fingertip_handles = [self.gym.find_asset_rigid_body_index(shadow_hand_asset, name) for name in self.fingertips]
        self.fingertip_another_handles = [self.gym.find_asset_rigid_body_index(shadow_hand_another_asset, name) for name in self.a_fingertips]

        # create fingertip force sensors, if needed
        if self.obs_type == "full_state" or self.obs_type == "pick_bottle" or self.asymmetric_obs:
            sensor_pose = gymapi.Transform()
            for ft_handle in self.fingertip_handles:
                self.gym.create_asset_force_sensor(shadow_hand_asset, ft_handle, sensor_pose)
            for ft_a_handle in self.fingertip_another_handles:
                self.gym.create_asset_force_sensor(shadow_hand_another_asset, ft_a_handle, sensor_pose)
        
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )
            if self.aggregate_mode >= 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # add hand - collision filter = -1 to use asset collision filters set in mjcf loader
            shadow_hand_actor = self.gym.create_actor(env_ptr, shadow_hand_asset, shadow_hand_start_pose, "hand", i, 0, 0)
            shadow_hand_another_actor = self.gym.create_actor(env_ptr, shadow_hand_another_asset, shadow_another_hand_start_pose, "another_hand", i, 0, 0)
            
            self.hand_start_states.append([shadow_hand_start_pose.p.x, shadow_hand_start_pose.p.y, shadow_hand_start_pose.p.z,
                                           shadow_hand_start_pose.r.x, shadow_hand_start_pose.r.y, shadow_hand_start_pose.r.z, shadow_hand_start_pose.r.w,
                                           0, 0, 0, 0, 0, 0])
            
            self.gym.set_actor_dof_properties(env_ptr, shadow_hand_actor, shadow_hand_dof_props)
            hand_idx = self.gym.get_actor_index(env_ptr, shadow_hand_actor, gymapi.DOMAIN_SIM)
            self.hand_indices.append(hand_idx)
            
            self.gym.set_actor_dof_properties(env_ptr, shadow_hand_another_actor, shadow_hand_another_dof_props)
            another_hand_idx = self.gym.get_actor_index(env_ptr, shadow_hand_another_actor, gymapi.DOMAIN_SIM)
            self.another_hand_indices.append(another_hand_idx)            

            # randomize colors and textures for rigid body
            num_bodies = self.gym.get_actor_rigid_body_count(env_ptr, shadow_hand_actor)
            hand_rigid_body_index = [[0,1,2,3], [4,5,6,7], [8,9,10,11], [12,13,14,15], [16,17,18,19,20], [21,22,23,24,25]]
            
            for n in self.agent_index[0]:
                colorx = random.uniform(0, 1)
                colory = random.uniform(0, 1)
                colorz = random.uniform(0, 1)
                for m in n:
                    for o in hand_rigid_body_index[m]:
                        self.gym.set_rigid_body_color(env_ptr, shadow_hand_actor, o, gymapi.MESH_VISUAL,
                                                gymapi.Vec3(colorx, colory, colorz))
            for n in self.agent_index[1]:                
                colorx = random.uniform(0, 1)
                colory = random.uniform(0, 1)
                colorz = random.uniform(0, 1)
                for m in n:
                    for o in hand_rigid_body_index[m]:
                        self.gym.set_rigid_body_color(env_ptr, shadow_hand_another_actor, o, gymapi.MESH_VISUAL,
                                                gymapi.Vec3(colorx, colory, colorz))
                # gym.set_rigid_body_texture(env, actor_handles[-1], n, gymapi.MESH_VISUAL,
                #                            loaded_texture_handle_list[random.randint(0, len(loaded_texture_handle_list)-1)])

            # create fingertip force-torque sensors
            if self.obs_type == "full_state" or self.obs_type == "pick_bottle" or self.asymmetric_obs:
                self.gym.enable_actor_dof_force_sensors(env_ptr, shadow_hand_actor)
                self.gym.enable_actor_dof_force_sensors(env_ptr, shadow_hand_another_actor)
            
            # add object

            object_handle1 = self.gym.create_actor(env_ptr, fixed_object_asset, object_start_pose1, "object", i+10000000, 0, 0)
            object_handle2 = self.gym.create_actor(env_ptr, moving_object_asset, object_start_pose2, "object", i, 0, 0)
            object_handle3 = self.gym.create_actor(env_ptr, fixed_object_asset, object_start_pose3, "object", i+10000000, 0, 0)
            object_handle4 = self.gym.create_actor(env_ptr, moving_object_asset, object_start_pose4, "object", i, 0, 0)
            object_handle5 = self.gym.create_actor(env_ptr, fixed_object_asset, object_start_pose5, "object", i+10000000, 0, 0)
            
            self.object1_init_state.append([object_start_pose1.p.x, object_start_pose1.p.y, object_start_pose1.p.z,
                                           object_start_pose1.r.x, object_start_pose1.r.y, object_start_pose1.r.z, object_start_pose1.r.w,
                                           0, 0, 0, 0, 0, 0])
            self.object2_init_state.append([object_start_pose2.p.x, object_start_pose2.p.y, object_start_pose2.p.z,
                                           object_start_pose2.r.x, object_start_pose2.r.y, object_start_pose2.r.z, object_start_pose2.r.w,
                                           0, 0, 0, 0, 0, 0])
            self.object3_init_state.append([object_start_pose3.p.x, object_start_pose3.p.y, object_start_pose3.p.z,
                                           object_start_pose3.r.x, object_start_pose3.r.y, object_start_pose3.r.z, object_start_pose3.r.w,
                                           0, 0, 0, 0, 0, 0])
            self.object4_init_state.append([object_start_pose4.p.x, object_start_pose4.p.y, object_start_pose4.p.z,
                                           object_start_pose4.r.x, object_start_pose4.r.y, object_start_pose4.r.z, object_start_pose4.r.w,
                                           0, 0, 0, 0, 0, 0])
            self.object5_init_state.append([object_start_pose5.p.x, object_start_pose5.p.y, object_start_pose5.p.z,
                                           object_start_pose5.r.x, object_start_pose5.r.y, object_start_pose5.r.z, object_start_pose5.r.w,
                                           0, 0, 0, 0, 0, 0])
            
            self.gym.set_actor_dof_properties(env_ptr, object_handle1, object_dof_props)
            self.gym.set_actor_dof_properties(env_ptr, object_handle2, object_dof_props)
            self.gym.set_actor_dof_properties(env_ptr, object_handle3, object_dof_props)
            self.gym.set_actor_dof_properties(env_ptr, object_handle4, object_dof_props)
            self.gym.set_actor_dof_properties(env_ptr, object_handle5, object_dof_props)
            object1_idx = self.gym.get_actor_index(env_ptr, object_handle1, gymapi.DOMAIN_SIM)
            object2_idx = self.gym.get_actor_index(env_ptr, object_handle2, gymapi.DOMAIN_SIM)
            object3_idx = self.gym.get_actor_index(env_ptr, object_handle3, gymapi.DOMAIN_SIM)
            object4_idx = self.gym.get_actor_index(env_ptr, object_handle4, gymapi.DOMAIN_SIM)
            object5_idx = self.gym.get_actor_index(env_ptr, object_handle5, gymapi.DOMAIN_SIM)
            self.object1_indices.append(object1_idx)
            self.object2_indices.append(object2_idx)
            self.object3_indices.append(object3_idx)
            self.object4_indices.append(object4_idx)
            self.object5_indices.append(object5_idx)

            # add table
            table_handle = self.gym.create_actor(env_ptr, table_asset, table_pose, "table", i, 0, 0)
            table_idx = self.gym.get_actor_index(env_ptr, table_handle, gymapi.DOMAIN_SIM)
            self.table_indices.append(table_idx)

            object_dof_props = self.gym.get_actor_dof_properties(env_ptr, object_handle1)
            for object_dof_prop in object_dof_props:
                object_dof_prop[4] = 100
                object_dof_prop[5] = 100
                object_dof_prop[6] = 5
                object_dof_prop[7] = 1
            self.gym.set_actor_dof_properties(env_ptr, object_handle1, object_dof_props)
            self.gym.set_actor_dof_properties(env_ptr, object_handle2, object_dof_props)
            self.gym.set_actor_dof_properties(env_ptr, object_handle3, object_dof_props)
            self.gym.set_actor_dof_properties(env_ptr, object_handle4, object_dof_props)
            self.gym.set_actor_dof_properties(env_ptr, object_handle5, object_dof_props)

            #set friction
            object1_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, object_handle1)
            for object1_shape_prop in object1_shape_props:
                object1_shape_prop.friction = 0.1
            object2_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, object_handle2)
            for object2_shape_prop in object2_shape_props:
                object2_shape_prop.friction = 100
            object3_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, object_handle3)
            for object3_shape_prop in object3_shape_props:
                object3_shape_prop.friction = 0.1
            object4_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, object_handle4)
            for object4_shape_prop in object4_shape_props:
                object4_shape_prop.friction = 100
            object5_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, object_handle5)
            for object5_shape_prop in object5_shape_props:
                object5_shape_prop.friction = 0.1
            self.gym.set_actor_rigid_shape_properties(env_ptr, object_handle1, object1_shape_props)
            self.gym.set_actor_rigid_shape_properties(env_ptr, object_handle2, object2_shape_props)
            self.gym.set_actor_rigid_shape_properties(env_ptr, object_handle3, object3_shape_props)
            self.gym.set_actor_rigid_shape_properties(env_ptr, object_handle4, object4_shape_props)
            self.gym.set_actor_rigid_shape_properties(env_ptr, object_handle5, object5_shape_props)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.shadow_hands.append(shadow_hand_actor)

        self.object1_init_state = to_torch(self.object1_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.object2_init_state = to_torch(self.object2_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.object3_init_state = to_torch(self.object3_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.object4_init_state = to_torch(self.object4_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.object5_init_state = to_torch(self.object5_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)

        self.hand_start_states = to_torch(self.hand_start_states, device=self.device).view(self.num_envs, 13)

        self.fingertip_handles = to_torch(self.fingertip_handles, dtype=torch.long, device=self.device)
        self.fingertip_another_handles = to_torch(self.fingertip_another_handles, dtype=torch.long, device=self.device)

        self.hand_indices = to_torch(self.hand_indices, dtype=torch.long, device=self.device)
        self.another_hand_indices = to_torch(self.another_hand_indices, dtype=torch.long, device=self.device)

        self.object1_indices = to_torch(self.object1_indices, dtype=torch.long, device=self.device)
        self.object2_indices = to_torch(self.object2_indices, dtype=torch.long, device=self.device)
        self.object3_indices = to_torch(self.object3_indices, dtype=torch.long, device=self.device)
        self.object4_indices = to_torch(self.object4_indices, dtype=torch.long, device=self.device)
        self.object5_indices = to_torch(self.object5_indices, dtype=torch.long, device=self.device)
        # self.goal_object_indices = to_torch(self.goal_object_indices, dtype=torch.long, device=self.device)
        self.table_indices = to_torch(self.table_indices, dtype=torch.long, device=self.device)

    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:], self.progress_buf[:], self.successes[:], self.consecutive_successes[:] = compute_hand_reward(
            self.rew_buf, self.reset_buf, self.progress_buf, self.successes, self.consecutive_successes,
            self.max_episode_length, 
            self.object1_pos, self.object1_rot, self.bottle1_cap_pos, 
            self.object2_pos, self.object2_rot, self.bottle2_cap_pos, 
            self.object3_pos, self.object3_rot, self.bottle3_cap_pos, 
            self.object4_pos, self.object4_rot, self.bottle4_cap_pos, 
            self.object5_pos, self.object5_rot, self.bottle5_cap_pos, 
            self.left_hand_pos, self.right_hand_pos, 
            self.right_hand_ff_pos, self.right_hand_mf_pos, self.right_hand_rf_pos, self.right_hand_lf_pos, self.right_hand_th_pos, 
            self.left_hand_ff_pos, self.left_hand_mf_pos, self.left_hand_rf_pos, self.left_hand_lf_pos, self.left_hand_th_pos,
            self.dist_reward_scale, self.rot_reward_scale, self.rot_eps, self.actions, self.action_penalty_scale,
            self.success_tolerance, self.fall_dist, self.fall_penalty,
            self.max_consecutive_successes, self.av_factor, (self.object_type == "pen")
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
        
        self.object1_pos = self.root_state_tensor[self.object1_indices, 0:3]
        self.object2_pos = self.root_state_tensor[self.object2_indices, 0:3]
        self.object3_pos = self.root_state_tensor[self.object3_indices, 0:3]
        self.object4_pos = self.root_state_tensor[self.object4_indices, 0:3]
        self.object5_pos = self.root_state_tensor[self.object5_indices, 0:3]

        self.cost_buf = torch.where(self.object2_pos[:, 1] < -0.73, torch.ones_like(self.cost_buf), self.cost_buf)
        self.cost_buf = torch.where(self.object2_pos[:, 1] > -0.67, torch.ones_like(self.cost_buf), self.cost_buf)
        self.cost_buf = torch.where(self.object4_pos[:, 1] < -0.53, torch.ones_like(self.cost_buf), self.cost_buf)
        self.cost_buf = torch.where(self.object4_pos[:, 1] > -0.47, torch.ones_like(self.cost_buf), self.cost_buf)
        self.cost_buf = torch.where(self.bottle2_bottom_pos[:, 1] < -0.73, torch.ones_like(self.cost_buf), self.cost_buf)
        self.cost_buf = torch.where(self.bottle2_bottom_pos[:, 1] > -0.67, torch.ones_like(self.cost_buf), self.cost_buf)
        self.cost_buf = torch.where(self.bottle4_bottom_pos[:, 1] < -0.53, torch.ones_like(self.cost_buf), self.cost_buf)
        self.cost_buf = torch.where(self.bottle4_bottom_pos[:, 1] > -0.47, torch.ones_like(self.cost_buf), self.cost_buf)
        self.cost_buf = torch.where(self.bottle2_cap_pos[:, 1] < -0.73, torch.ones_like(self.cost_buf), self.cost_buf)
        self.cost_buf = torch.where(self.bottle2_cap_pos[:, 1] > -0.67, torch.ones_like(self.cost_buf), self.cost_buf)
        self.cost_buf = torch.where(self.bottle4_cap_pos[:, 1] < -0.53, torch.ones_like(self.cost_buf), self.cost_buf)
        self.cost_buf = torch.where(self.bottle4_cap_pos[:, 1] > -0.47, torch.ones_like(self.cost_buf), self.cost_buf)

        return self.cost_buf

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        if self.obs_type == "full_state" or self.obs_type == "pick_bottle" or self.asymmetric_obs:
            self.gym.refresh_force_sensor_tensor(self.sim)
            self.gym.refresh_dof_force_tensor(self.sim)

        self.object1_pose = self.root_state_tensor[self.object1_indices, 0:7]
        self.object1_pos = self.root_state_tensor[self.object1_indices, 0:3]
        self.object1_rot = self.root_state_tensor[self.object1_indices, 3:7]
        self.object1_linvel = self.root_state_tensor[self.object1_indices, 7:10]
        self.object1_angvel = self.root_state_tensor[self.object1_indices, 10:13]
        
        self.object2_pose = self.root_state_tensor[self.object2_indices, 0:7]
        self.object2_pos = self.root_state_tensor[self.object2_indices, 0:3]
        self.object2_rot = self.root_state_tensor[self.object2_indices, 3:7]
        self.object2_linvel = self.root_state_tensor[self.object2_indices, 7:10]
        self.object2_angvel = self.root_state_tensor[self.object2_indices, 10:13]
        
        self.object3_pose = self.root_state_tensor[self.object3_indices, 0:7]
        self.object3_pos = self.root_state_tensor[self.object3_indices, 0:3]
        self.object3_rot = self.root_state_tensor[self.object3_indices, 3:7]
        self.object3_linvel = self.root_state_tensor[self.object3_indices, 7:10]
        self.object3_angvel = self.root_state_tensor[self.object3_indices, 10:13]
        
        self.object4_pose = self.root_state_tensor[self.object4_indices, 0:7]
        self.object4_pos = self.root_state_tensor[self.object4_indices, 0:3]
        self.object4_rot = self.root_state_tensor[self.object4_indices, 3:7]
        self.object4_linvel = self.root_state_tensor[self.object4_indices, 7:10]
        self.object4_angvel = self.root_state_tensor[self.object4_indices, 10:13]
        
        self.object5_pose = self.root_state_tensor[self.object5_indices, 0:7]
        self.object5_pos = self.root_state_tensor[self.object5_indices, 0:3]
        self.object5_rot = self.root_state_tensor[self.object5_indices, 3:7]
        self.object5_linvel = self.root_state_tensor[self.object5_indices, 7:10]
        self.object5_angvel = self.root_state_tensor[self.object5_indices, 10:13]

        self.bottle1_cap_up = self.rigid_body_states[:, 26 * 2 + 4*1-1, 0:3].clone()
        self.bottle2_cap_up = self.rigid_body_states[:, 26 * 2 + 4*2-1, 0:3].clone()
        self.bottle3_cap_up = self.rigid_body_states[:, 26 * 2 + 4*3-1, 0:3].clone()
        self.bottle4_cap_up = self.rigid_body_states[:, 26 * 2 + 4*4-1, 0:3].clone()
        self.bottle5_cap_up = self.rigid_body_states[:, 26 * 2 + 4*5-1, 0:3].clone()

        self.bottle1_cap_pos = self.object1_pos + quat_apply(self.object1_rot, to_torch([0, 1, 0], device=self.device).repeat(self.num_envs, 1) * -0.0)
        self.bottle1_cap_pos = self.bottle1_cap_pos + quat_apply(self.object1_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.15)
        self.bottle2_cap_pos = self.object2_pos + quat_apply(self.object2_rot, to_torch([0, 1, 0], device=self.device).repeat(self.num_envs, 1) * -0.0)
        self.bottle2_cap_pos = self.bottle2_cap_pos + quat_apply(self.object2_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.15)
        self.bottle3_cap_pos = self.object3_pos + quat_apply(self.object3_rot, to_torch([0, 1, 0], device=self.device).repeat(self.num_envs, 1) * -0.0)
        self.bottle3_cap_pos = self.bottle3_cap_pos + quat_apply(self.object3_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.15)
        self.bottle4_cap_pos = self.object4_pos + quat_apply(self.object4_rot, to_torch([0, 1, 0], device=self.device).repeat(self.num_envs, 1) * -0.0)
        self.bottle4_cap_pos = self.bottle4_cap_pos + quat_apply(self.object4_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.15)
        self.bottle5_cap_pos = self.object5_pos + quat_apply(self.object5_rot, to_torch([0, 1, 0], device=self.device).repeat(self.num_envs, 1) * -0.0)
        self.bottle5_cap_pos = self.bottle5_cap_pos + quat_apply(self.object5_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.15)

        self.bottle1_bottom_pos = self.object1_pos + quat_apply(self.object1_rot, to_torch([0, 1, 0], device=self.device).repeat(self.num_envs, 1) * -0.0)
        self.bottle1_bottom_pos = self.bottle1_bottom_pos + quat_apply(self.object1_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * -0.15)
        self.bottle2_bottom_pos = self.object2_pos + quat_apply(self.object2_rot, to_torch([0, 1, 0], device=self.device).repeat(self.num_envs, 1) * -0.0)
        self.bottle2_bottom_pos = self.bottle2_bottom_pos + quat_apply(self.object2_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * -0.15)
        self.bottle3_bottom_pos = self.object3_pos + quat_apply(self.object3_rot, to_torch([0, 1, 0], device=self.device).repeat(self.num_envs, 1) * -0.0)
        self.bottle3_bottom_pos = self.bottle3_bottom_pos + quat_apply(self.object3_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * -0.15)
        self.bottle4_bottom_pos = self.object4_pos + quat_apply(self.object4_rot, to_torch([0, 1, 0], device=self.device).repeat(self.num_envs, 1) * -0.0)
        self.bottle4_bottom_pos = self.bottle4_bottom_pos + quat_apply(self.object4_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * -0.15)
        self.bottle5_bottom_pos = self.object5_pos + quat_apply(self.object5_rot, to_torch([0, 1, 0], device=self.device).repeat(self.num_envs, 1) * -0.0)
        self.bottle5_bottom_pos = self.bottle5_bottom_pos + quat_apply(self.object5_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * -0.15)

        self.left_hand_pos = self.rigid_body_states[:, 3 + 26, 0:3]
        self.left_hand_rot = self.rigid_body_states[:, 3 + 26, 3:7]
        self.left_hand_pos = self.left_hand_pos + quat_apply(self.left_hand_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.08)
        self.left_hand_pos = self.left_hand_pos + quat_apply(self.left_hand_rot, to_torch([0, 1, 0], device=self.device).repeat(self.num_envs, 1) * -0.02)

        self.right_hand_pos = self.rigid_body_states[:, 3, 0:3]
        self.right_hand_rot = self.rigid_body_states[:, 3, 3:7]
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
        self.left_hand_ff_pos = self.rigid_body_states[:, 26+7, 0:3]
        self.left_hand_ff_rot = self.rigid_body_states[:, 26+7, 3:7]
        self.left_hand_ff_pos = self.left_hand_ff_pos + quat_apply(self.left_hand_ff_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)
        self.left_hand_mf_pos = self.rigid_body_states[:, 26+11, 0:3]
        self.left_hand_mf_rot = self.rigid_body_states[:, 26+11, 3:7]
        self.left_hand_mf_pos = self.left_hand_mf_pos + quat_apply(self.left_hand_mf_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)
        self.left_hand_rf_pos = self.rigid_body_states[:, 26+15, 0:3]
        self.left_hand_rf_rot = self.rigid_body_states[:, 26+15, 3:7]
        self.left_hand_rf_pos = self.left_hand_rf_pos + quat_apply(self.left_hand_rf_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)
        self.left_hand_lf_pos = self.rigid_body_states[:, 26+20, 0:3]
        self.left_hand_lf_rot = self.rigid_body_states[:, 26+20, 3:7]
        self.left_hand_lf_pos = self.left_hand_lf_pos + quat_apply(self.left_hand_lf_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)
        self.left_hand_th_pos = self.rigid_body_states[:, 26+25, 0:3]
        self.left_hand_th_rot = self.rigid_body_states[:, 26+25, 3:7]
        self.left_hand_th_pos = self.left_hand_th_pos + quat_apply(self.left_hand_th_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)



        self.fingertip_state = self.rigid_body_states[:, self.fingertip_handles][:, :, 0:13]
        self.fingertip_pos = self.rigid_body_states[:, self.fingertip_handles][:, :, 0:3]
        self.fingertip_another_state = self.rigid_body_states[:, self.fingertip_another_handles][:, :, 0:13]
        self.fingertip_another_pos = self.rigid_body_states[:, self.fingertip_another_handles][:, :, 0:3]

        self.compute_full_state()

        if self.asymmetric_obs:
            self.compute_full_state(True)

    def compute_full_state(self, asymm_obs=False):
        num_ft_states = 13 * int(self.num_fingertips / 2)  # 65
        num_ft_force_torques = 6 * int(self.num_fingertips / 2)  # 30

        self.obs_buf[:, 0:self.num_shadow_hand_dofs] = unscale(self.shadow_hand_dof_pos,
                                                            self.shadow_hand_dof_lower_limits, self.shadow_hand_dof_upper_limits)
        self.obs_buf[:, self.num_shadow_hand_dofs:2*self.num_shadow_hand_dofs] = self.vel_obs_scale * self.shadow_hand_dof_vel
        self.obs_buf[:, 2*self.num_shadow_hand_dofs:3*self.num_shadow_hand_dofs] = self.force_torque_obs_scale * self.dof_force_tensor[:, :24]

        fingertip_obs_start = 72  # 24*3
        self.obs_buf[:, fingertip_obs_start:fingertip_obs_start + num_ft_states] = self.fingertip_state.reshape(self.num_envs, num_ft_states)
        self.obs_buf[:, fingertip_obs_start + num_ft_states:fingertip_obs_start + num_ft_states +
                    num_ft_force_torques] = self.force_torque_obs_scale * self.vec_sensor_tensor[:, :30]
        
        hand_pose_start = fingertip_obs_start + 95
        self.obs_buf[:, hand_pose_start:hand_pose_start + 3] = self.right_hand_pos
        self.obs_buf[:, hand_pose_start+3:hand_pose_start+4] = get_euler_xyz(self.hand_orientations[self.hand_indices, :])[0].unsqueeze(-1)
        self.obs_buf[:, hand_pose_start+4:hand_pose_start+5] = get_euler_xyz(self.hand_orientations[self.hand_indices, :])[1].unsqueeze(-1)
        self.obs_buf[:, hand_pose_start+5:hand_pose_start+6] = get_euler_xyz(self.hand_orientations[self.hand_indices, :])[2].unsqueeze(-1)

        action_obs_start = hand_pose_start + 6
        self.obs_buf[:, action_obs_start:action_obs_start + 26] = self.actions[:, :26]

        # another_hand
        another_hand_start = action_obs_start + 26
        self.obs_buf[:, another_hand_start:self.num_shadow_hand_dofs + another_hand_start] = unscale(self.shadow_hand_another_dof_pos,
                                                            self.shadow_hand_dof_lower_limits, self.shadow_hand_dof_upper_limits)
        self.obs_buf[:, self.num_shadow_hand_dofs + another_hand_start:2*self.num_shadow_hand_dofs + another_hand_start] = self.vel_obs_scale * self.shadow_hand_another_dof_vel
        self.obs_buf[:, 2*self.num_shadow_hand_dofs + another_hand_start:3*self.num_shadow_hand_dofs + another_hand_start] = self.force_torque_obs_scale * self.dof_force_tensor[:, 24:48]

        fingertip_another_obs_start = another_hand_start + 72
        self.obs_buf[:, fingertip_another_obs_start:fingertip_another_obs_start + num_ft_states] = self.fingertip_another_state.reshape(self.num_envs, num_ft_states)
        self.obs_buf[:, fingertip_another_obs_start + num_ft_states:fingertip_another_obs_start + num_ft_states +
                    num_ft_force_torques] = self.force_torque_obs_scale * self.vec_sensor_tensor[:, 30:]

        hand_another_pose_start = fingertip_another_obs_start + 95
        self.obs_buf[:, hand_another_pose_start:hand_another_pose_start + 3] = self.left_hand_pos
        self.obs_buf[:, hand_another_pose_start+3:hand_another_pose_start+4] = get_euler_xyz(self.hand_orientations[self.another_hand_indices, :])[0].unsqueeze(-1)
        self.obs_buf[:, hand_another_pose_start+4:hand_another_pose_start+5] = get_euler_xyz(self.hand_orientations[self.another_hand_indices, :])[1].unsqueeze(-1)
        self.obs_buf[:, hand_another_pose_start+5:hand_another_pose_start+6] = get_euler_xyz(self.hand_orientations[self.another_hand_indices, :])[2].unsqueeze(-1)

        action_another_obs_start = hand_another_pose_start + 6
        self.obs_buf[:, action_another_obs_start:action_another_obs_start + 26] = self.actions[:, 26:]

        obj_obs_start = action_another_obs_start + 26  # 144
        self.obs_buf[:, obj_obs_start:obj_obs_start + 7] = self.object2_pose
        self.obs_buf[:, obj_obs_start + 7:obj_obs_start + 10] = self.object2_linvel
        self.obs_buf[:, obj_obs_start + 10:obj_obs_start + 13] = self.vel_obs_scale * self.object2_angvel
        self.obs_buf[:, obj_obs_start + 13:obj_obs_start + 16] = self.bottle2_cap_pos
        

        another_obj_obs_start = obj_obs_start + 13  # 144
        self.obs_buf[:, another_obj_obs_start:another_obj_obs_start + 7] = self.object4_pose
        self.obs_buf[:, another_obj_obs_start + 7:another_obj_obs_start + 10] = self.object4_linvel
        self.obs_buf[:, another_obj_obs_start + 10:another_obj_obs_start + 13] = self.vel_obs_scale * self.object4_angvel
        self.obs_buf[:, another_obj_obs_start + 13:another_obj_obs_start + 16] = self.bottle4_cap_pos



    def reset(self, env_ids):
        # randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        # generate random values
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), self.num_shadow_hand_dofs * 2 + 5), device=self.device)

        # reset object
        self.root_state_tensor[self.object1_indices[env_ids]] = self.object1_init_state[env_ids].clone()
        self.root_state_tensor[self.object2_indices[env_ids]] = self.object2_init_state[env_ids].clone()
        self.root_state_tensor[self.object3_indices[env_ids]] = self.object3_init_state[env_ids].clone()
        self.root_state_tensor[self.object4_indices[env_ids]] = self.object4_init_state[env_ids].clone()
        self.root_state_tensor[self.object5_indices[env_ids]] = self.object5_init_state[env_ids].clone()
        
        self.root_state_tensor[self.object1_indices[env_ids], 0:2] = self.object1_init_state[env_ids, 0:2] + \
            self.reset_position_noise * rand_floats[:, 0:2]
        self.root_state_tensor[self.object2_indices[env_ids], 0:2] = self.object2_init_state[env_ids, 0:2] + \
            self.reset_position_noise * rand_floats[:, 0:2]
        self.root_state_tensor[self.object3_indices[env_ids], 0:2] = self.object3_init_state[env_ids, 0:2] + \
            self.reset_position_noise * rand_floats[:, 0:2]
        self.root_state_tensor[self.object4_indices[env_ids], 0:2] = self.object4_init_state[env_ids, 0:2] + \
            self.reset_position_noise * rand_floats[:, 0:2]
        self.root_state_tensor[self.object5_indices[env_ids], 0:2] = self.object5_init_state[env_ids, 0:2] + \
            self.reset_position_noise * rand_floats[:, 0:2]
        self.root_state_tensor[self.object1_indices[env_ids], self.up_axis_idx] = self.object1_init_state[env_ids, self.up_axis_idx] + \
            self.reset_position_noise * rand_floats[:, self.up_axis_idx]
        self.root_state_tensor[self.object2_indices[env_ids], self.up_axis_idx] = self.object2_init_state[env_ids, self.up_axis_idx] + \
            self.reset_position_noise * rand_floats[:, self.up_axis_idx]
        self.root_state_tensor[self.object3_indices[env_ids], self.up_axis_idx] = self.object3_init_state[env_ids, self.up_axis_idx] + \
            self.reset_position_noise * rand_floats[:, self.up_axis_idx]
        self.root_state_tensor[self.object4_indices[env_ids], self.up_axis_idx] = self.object4_init_state[env_ids, self.up_axis_idx] + \
            self.reset_position_noise * rand_floats[:, self.up_axis_idx]
        self.root_state_tensor[self.object5_indices[env_ids], self.up_axis_idx] = self.object5_init_state[env_ids, self.up_axis_idx] + \
            self.reset_position_noise * rand_floats[:, self.up_axis_idx]

        self.root_state_tensor[self.object1_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.object1_indices[env_ids], 7:13])
        self.root_state_tensor[self.object2_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.object2_indices[env_ids], 7:13])
        self.root_state_tensor[self.object3_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.object3_indices[env_ids], 7:13])
        self.root_state_tensor[self.object4_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.object4_indices[env_ids], 7:13])
        self.root_state_tensor[self.object5_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.object5_indices[env_ids], 7:13])
        
        object1_indices = torch.unique(torch.cat([self.object1_indices[env_ids]]))
        object2_indices = torch.unique(torch.cat([self.object2_indices[env_ids]]))
        object3_indices = torch.unique(torch.cat([self.object3_indices[env_ids]]))
        object4_indices = torch.unique(torch.cat([self.object4_indices[env_ids]]))
        object5_indices = torch.unique(torch.cat([self.object5_indices[env_ids]]))

        # reset shadow hand
        delta_max = self.shadow_hand_dof_upper_limits - self.shadow_hand_dof_default_pos
        delta_min = self.shadow_hand_dof_lower_limits - self.shadow_hand_dof_default_pos
        rand_delta = delta_min + (delta_max - delta_min) * rand_floats[:, 5:5+self.num_shadow_hand_dofs]

        pos = self.shadow_hand_default_dof_pos + self.reset_dof_pos_noise * rand_delta

        self.shadow_hand_dof_pos[env_ids, :] = pos
        self.shadow_hand_another_dof_pos[env_ids, :] = pos

        self.shadow_hand_dof_vel[env_ids, :] = self.shadow_hand_dof_default_vel + \
            self.reset_dof_vel_noise * rand_floats[:, 5+self.num_shadow_hand_dofs:5+self.num_shadow_hand_dofs*2]   

        self.shadow_hand_another_dof_vel[env_ids, :] = self.shadow_hand_dof_default_vel + \
            self.reset_dof_vel_noise * rand_floats[:, 5+self.num_shadow_hand_dofs:5+self.num_shadow_hand_dofs*2]

        self.prev_targets[env_ids, :self.num_shadow_hand_dofs] = pos
        self.cur_targets[env_ids, :self.num_shadow_hand_dofs] = pos

        self.prev_targets[env_ids, self.num_shadow_hand_dofs:self.num_shadow_hand_dofs*2] = pos
        self.cur_targets[env_ids, self.num_shadow_hand_dofs:self.num_shadow_hand_dofs*2] = pos

        # self.prev_targets[env_ids, self.num_shadow_hand_dofs*2:self.num_shadow_hand_dofs*2 + 2] = to_torch([0, 0], device=self.device)
        # self.cur_targets[env_ids, self.num_shadow_hand_dofs*2:self.num_shadow_hand_dofs*2 + 2] = to_torch([0, 0], device=self.device)
# 
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
                                              object1_indices,
                                              object2_indices, 
                                              object3_indices, 
                                              object4_indices, 
                                              object5_indices,
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
            self.cur_targets[:, self.actuated_dof_indices] = scale(self.actions[:, 6:26],
                                                                   self.shadow_hand_dof_lower_limits[self.actuated_dof_indices], self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])
            self.cur_targets[:, self.actuated_dof_indices] = self.act_moving_average * self.cur_targets[:,
                                                                                                        self.actuated_dof_indices] + (1.0 - self.act_moving_average) * self.prev_targets[:, self.actuated_dof_indices]
            self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(self.cur_targets[:, self.actuated_dof_indices],
                                                                          self.shadow_hand_dof_lower_limits[self.actuated_dof_indices], self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])

            self.cur_targets[:, self.actuated_dof_indices + 24] = scale(self.actions[:, 32:52],
                                                                   self.shadow_hand_dof_lower_limits[self.actuated_dof_indices], self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])
            self.cur_targets[:, self.actuated_dof_indices + 24] = self.act_moving_average * self.cur_targets[:,
                                                                                                        self.actuated_dof_indices + 24] + (1.0 - self.act_moving_average) * self.prev_targets[:, self.actuated_dof_indices]
            self.cur_targets[:, self.actuated_dof_indices + 24] = tensor_clamp(self.cur_targets[:, self.actuated_dof_indices + 24],
                                                                          self.shadow_hand_dof_lower_limits[self.actuated_dof_indices], self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])
            

            self.apply_forces[:, 1, :] = actions[:, 0:3] * self.dt * self.transition_scale * 100000
            self.apply_forces[:, 1 + 26, :] = actions[:, 26:29] * self.dt * self.transition_scale * 100000
            self.apply_torque[:, 1, :] = self.actions[:, 3:6] * self.dt * self.orientation_scale * 1000
            self.apply_torque[:, 1 + 26, :] = self.actions[:, 29:32] * self.dt * self.orientation_scale * 1000   

            self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.apply_forces), gymtorch.unwrap_tensor(self.apply_torque), gymapi.ENV_SPACE)

        self.prev_targets[:, self.actuated_dof_indices] = self.cur_targets[:, self.actuated_dof_indices]
        self.prev_targets[:, self.actuated_dof_indices + 24] = self.cur_targets[:, self.actuated_dof_indices + 24]
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.cur_targets))


    def post_physics_step(self):
        self.progress_buf += 1
        self.randomize_buf += 1

        self.compute_observations()
        self.compute_reward(self.actions)
        self.compute_cost()

        if self.viewer and self.debug_viz:
            # draw axes on target object
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            for i in range(self.num_envs):
                # self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], targetx[0], targetx[1], targetx[2]], [0.85, 0.1, 0.1])
                # self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], targety[0], targety[1], targety[2]], [0.1, 0.85, 0.1])
                # self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], targetz[0], targetz[1], targetz[2]], [0.1, 0.1, 0.85])

                # objectx = (self.object_pos[i] + quat_apply(self.object_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                # objecty = (self.object_pos[i] + quat_apply(self.object_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                # objectz = (self.object_pos[i] + quat_apply(self.object_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                # p0 = self.object_pos[i].cpu().numpy()
                # self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], objectx[0], objectx[1], objectx[2]], [0.85, 0.1, 0.1])
                # self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], objecty[0], objecty[1], objecty[2]], [0.1, 0.85, 0.1])
                # self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], objectz[0], objectz[1], objectz[2]], [0.1, 0.1, 0.85])

                bottle_cap_posx = (self.bottle_cap_pos[i] + quat_apply(self.object_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                bottle_cap_posy = (self.bottle_cap_pos[i] + quat_apply(self.object_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                bottle_cap_posz = (self.bottle_cap_pos[i] + quat_apply(self.object_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.bottle_cap_pos[i].cpu().numpy()
                # self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], bottle_cap_posx[0], bottle_cap_posx[1], bottle_cap_posx[2]], [0.85, 0.1, 0.1])
                # self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], bottle_cap_posy[0], bottle_cap_posy[1], bottle_cap_posy[2]], [0.1, 0.85, 0.1])
                # self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], bottle_cap_posz[0], bottle_cap_posz[1], bottle_cap_posz[2]], [0.1, 0.1, 0.85])
                
                bottle_posx = (self.bottle_pos[i] + quat_apply(self.object_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                bottle_posy = (self.bottle_pos[i] + quat_apply(self.object_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                bottle_posz = (self.bottle_pos[i] + quat_apply(self.object_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.bottle_pos[i].cpu().numpy()
                # self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], bottle_posx[0], bottle_posx[1], bottle_posx[2]], [0.85, 0.1, 0.1])
                # self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], bottle_posy[0], bottle_posy[1], bottle_posy[2]], [0.1, 0.85, 0.1])
                # self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], bottle_posz[0], bottle_posz[1], bottle_posz[2]], [0.1, 0.1, 0.85])

                left_hand_posx = (self.left_hand_pos[i] + quat_apply(self.left_hand_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                left_hand_posy = (self.left_hand_pos[i] + quat_apply(self.left_hand_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                left_hand_posz = (self.left_hand_pos[i] + quat_apply(self.left_hand_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.left_hand_pos[i].cpu().numpy()
                # self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], left_hand_posx[0], left_hand_posx[1], left_hand_posx[2]], [0.85, 0.1, 0.1])
                # self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], left_hand_posy[0], left_hand_posy[1], left_hand_posy[2]], [0.1, 0.85, 0.1])
                # self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], left_hand_posz[0], left_hand_posz[1], left_hand_posz[2]], [0.1, 0.1, 0.85])

                # right_hand_posx = (self.right_hand_pos[i] + quat_apply(self.right_hand_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                # right_hand_posy = (self.right_hand_pos[i] + quat_apply(self.right_hand_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                # right_hand_posz = (self.right_hand_pos[i] + quat_apply(self.right_hand_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                # p0 = self.right_hand_pos[i].cpu().numpy()
                # self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], right_hand_posx[0], right_hand_posx[1], right_hand_posx[2]], [0.85, 0.1, 0.1])
                # self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], right_hand_posy[0], right_hand_posy[1], right_hand_posy[2]], [0.1, 0.85, 0.1])
                # self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], right_hand_posz[0], right_hand_posz[1], right_hand_posz[2]], [0.1, 0.1, 0.85])

                # self.add_debug_lines(self.envs[i], self.right_hand_ff_pos[i], self.right_hand_ff_rot[i])
                # self.add_debug_lines(self.envs[i], self.right_hand_mf_pos[i], self.right_hand_mf_rot[i])
                # self.add_debug_lines(self.envs[i], self.right_hand_rf_pos[i], self.right_hand_rf_rot[i])
                # self.add_debug_lines(self.envs[i], self.right_hand_lf_pos[i], self.right_hand_lf_rot[i])
                # self.add_debug_lines(self.envs[i], self.right_hand_th_pos[i], self.right_hand_th_rot[i])

    def add_debug_lines(self, env, pos, rot):
        posx = (pos + quat_apply(rot, to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
        posy = (pos + quat_apply(rot, to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
        posz = (pos + quat_apply(rot, to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

        p0 = pos.cpu().numpy()
        self.gym.add_lines(self.viewer, env, 1, [p0[0], p0[1], p0[2], posx[0], posx[1], posx[2]], [0.85, 0.1, 0.1])
        self.gym.add_lines(self.viewer, env, 1, [p0[0], p0[1], p0[2], posy[0], posy[1], posy[2]], [0.1, 0.85, 0.1])
        self.gym.add_lines(self.viewer, env, 1, [p0[0], p0[1], p0[2], posz[0], posz[1], posz[2]], [0.1, 0.1, 0.85])


#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_hand_reward(
    rew_buf, reset_buf, progress_buf, successes, consecutive_successes,
    max_episode_length: float,
    object1_pos, object1_rot, bottle1_cap_pos, 
    object2_pos, object2_rot, bottle2_cap_pos,
    object3_pos, object3_rot, bottle3_cap_pos,
    object4_pos, object4_rot, bottle4_cap_pos,
    object5_pos, object5_rot, bottle5_cap_pos,
    left_hand_pos, right_hand_pos, 
    right_hand_ff_pos, right_hand_mf_pos, right_hand_rf_pos, right_hand_lf_pos, right_hand_th_pos,
    left_hand_ff_pos, left_hand_mf_pos, left_hand_rf_pos, left_hand_lf_pos, left_hand_th_pos,
    dist_reward_scale: float, rot_reward_scale: float, rot_eps: float,
    actions, action_penalty_scale: float,
    success_tolerance: float, fall_dist: float,
    fall_penalty: float, max_consecutive_successes: int, av_factor: float, ignore_z_rot: bool
):

    right_hand_dist = torch.norm(bottle2_cap_pos - right_hand_pos, p=2, dim=-1)
    left_hand_dist = torch.norm(bottle4_cap_pos - left_hand_pos, p=2, dim=-1)

    right_hand_finger_dist = (torch.norm(bottle2_cap_pos - right_hand_ff_pos, p=2, dim=-1) + torch.norm(bottle2_cap_pos - right_hand_mf_pos, p=2, dim=-1)
                            + torch.norm(bottle2_cap_pos - right_hand_rf_pos, p=2, dim=-1) + torch.norm(bottle2_cap_pos - right_hand_lf_pos, p=2, dim=-1) 
                            + torch.norm(bottle2_cap_pos - right_hand_th_pos, p=2, dim=-1))
    left_hand_finger_dist = (torch.norm(bottle4_cap_pos - left_hand_ff_pos, p=2, dim=-1) + torch.norm(bottle4_cap_pos - left_hand_mf_pos, p=2, dim=-1)
                            + torch.norm(bottle4_cap_pos - left_hand_rf_pos, p=2, dim=-1) + torch.norm(bottle4_cap_pos - left_hand_lf_pos, p=2, dim=-1) 
                            + torch.norm(bottle4_cap_pos - left_hand_th_pos, p=2, dim=-1))
    
    right_hand_dist_rew = -right_hand_finger_dist
    left_hand_dist_rew = -left_hand_finger_dist

    action_penalty = torch.sum(actions ** 2, dim=-1)

    # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
    # reward = torch.exp(-0.05*(up_rew * dist_reward_scale)) + torch.exp(-0.05*(right_hand_dist_rew * dist_reward_scale)) + torch.exp(-0.05*(left_hand_dist_rew * dist_reward_scale))
    up_bottle_2_rew = torch.zeros_like(right_hand_dist_rew)
    up_bottle_4_rew = torch.zeros_like(right_hand_dist_rew)

    up_bottle_2_rew =  (object2_pos[:, 2]-0.46)*20
    up_bottle_4_rew =  (object4_pos[:, 2]-0.46)*20
    up_rew = torch.minimum(up_bottle_2_rew, up_bottle_4_rew)

    
    
    resets = torch.where(object1_pos[:, 2] <= 0.45, torch.ones_like(reset_buf), reset_buf)
    resets = torch.where(object2_pos[:, 2] <= 0.45, torch.ones_like(resets), resets)
    resets = torch.where(object3_pos[:, 2] <= 0.45, torch.ones_like(resets), resets)
    resets = torch.where(object4_pos[:, 2] <= 0.45, torch.ones_like(resets), resets)
    resets = torch.where(object5_pos[:, 2] <= 0.45, torch.ones_like(resets), resets)
    resets = torch.where(right_hand_finger_dist >= 0.7, torch.ones_like(resets), resets)
    resets = torch.where(left_hand_finger_dist >= 0.7, torch.ones_like(resets), resets)
    
    reward = right_hand_dist_rew + left_hand_dist_rew + up_rew - resets
    print()
    print("righthand", right_hand_dist_rew[0])
    print("lefthand", left_hand_dist_rew[0])
    print("up2", up_bottle_2_rew[0])
    print("up4", up_bottle_4_rew[0])
    print("total", reward[0])
    print("resets", -resets[0])

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

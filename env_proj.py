from typing import Callable, Optional, List, Dict, Any
from collections import namedtuple
import os
from isaacgym import gymapi, gymtorch
import torch
import utils
from utils import heading_zup, axang2quat, rotatepoint, quatconj, quatmultiply, quatdiff_normalized, quat_inverse, calc_heading_quat, to_torch
from poselib.core import quat_mul
from isaacgym import gymutil
from env import ICCGANHumanoidVRControl
import random
from ref_motion import ReferenceMotion
import numpy as np

def parse_kwarg(kwargs: dict, key: str, default_val: Any):
    return kwargs[key] if key in kwargs else default_val

class DiscriminatorConfig(object):
    def __init__(self,
        key_links: Optional[List[str]]=None, ob_horizon: Optional[int]=None, 
        parent_link: Optional[str]=None, local_pos: bool=False, local_height: bool=True,
        replay_speed: Optional[Callable]=None, motion_file: Optional[str]=None,
        weight:Optional[float]=None
    ):
        self.motion_file = motion_file
        self.key_links = key_links
        self.parent_link = parent_link
        self.local_pos = local_pos
        self.local_height = local_height
        self.replay_speed = replay_speed
        self.ob_horizon = ob_horizon
        self.weight = weight

PERTURB_OBJS = [
    ["small", 50],
    ["small", 20],
]
DiscriminatorProperty = namedtuple("DiscriminatorProperty",
    "name key_links parent_link local_pos local_height replay_speed ob_horizon id"
)

class SensorInputConfig(object):
    def __init__(self,
              rlh_localPos: Optional[str]=None,  
              rlh_localRot: Optional[str]=None, 
              joystick: Optional[str]=None):
        
        self.rlh_localPos = rlh_localPos
        self.rlh_localRot = rlh_localRot
        self.joystick = joystick
        
SensorInputProperty = namedtuple("SensorInputProperty",
    "name rlh_localPos rlh_localRot joystick")

class ICCGANHumanoidProjectile(ICCGANHumanoidVRControl):
    STRIKE_BODY_NAMES = ["right_hand", "right_lower_arm"]

    def __init__(self, *args, 
                sensor_inputs: Optional[Dict[str, SensorInputConfig]]=None,
                 **kwargs):

        self.sensor_inputs = sensor_inputs
        self.strike_body_names = parse_kwarg(kwargs, "strike_body_names", self.STRIKE_BODY_NAMES)
        super().__init__(*args, sensor_inputs = self.sensor_inputs, **kwargs)

        self._calc_perturb_times()
        self._proj_dist_min = 4
        self._proj_dist_max = 5
        self._proj_h_min = 1
        self._proj_h_max = 2
        self._proj_steps = 150
        self._proj_warmup_steps = 1
        self._proj_speed_min = 10
        self._proj_speed_max = 15
        self.perterb = False

        dir_body = ["right_hand", "left_hand"]
        tar_body = ["right_lower_arm", "left_lower_arm"]
        self.dir_body_idx = []
        self.tar_body_idx = []
        for i in range(len(tar_body)):
            self.dir_body_idx.append(self.gym.find_actor_rigid_body_handle(self.envs[0], self.actors[0], dir_body[i]))
            self.tar_body_idx.append(self.gym.find_actor_rigid_body_handle(self.envs[0], self.actors[0], tar_body[i]))

    def create_envs(self, n: int):
        #! projectile assets 
        self._proj_handles = []
        self._load_proj_asset()

        if self.control_mode == "position":
            control_mode = gymapi.DOF_MODE_POS
        elif self.control_mode == "torque":
            control_mode = gymapi.DOF_MODE_EFFORT
        else:
            control_mode = gymapi.DOF_MODE_NONE

        envs, humanoid_handles = [], []
        env_spacing = 3

        asset_options = gymapi.AssetOptions()
        # asset_options.fix_base_link = True
        asset_options.angular_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.default_dof_drive_mode = int(gymapi.DOF_MODE_NONE)
        actor_asset = self.gym.load_asset(self.sim, os.path.abspath(os.path.dirname(self.character_model)), os.path.basename(self.character_model), asset_options)
        spacing_lower = gymapi.Vec3(-env_spacing, -env_spacing, 0)
        spacing_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
        n_envs_per_row = int(n**0.5)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.vector_up(0.89))
        start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        
        self.num_bodies = self.gym.get_asset_rigid_body_count(actor_asset)
        self.num_dof = self.gym.get_asset_dof_count(actor_asset)
        self.num_joints = self.gym.get_asset_joint_count(actor_asset)

        for i in range(n):
            env = self.gym.create_env(self.sim, spacing_lower, spacing_upper, n_envs_per_row)
            humanoid_handle = self.gym.create_actor(env, actor_asset, start_pose, "actor", i, -1, 0)
            humanoid_handles.append(humanoid_handle)
            # enable PD controlget_asset_dof_properties
            # Kp (stiffness) and Kd (damping) are defined inside the mjcf xml file
            dof_prop = self.gym.get_asset_dof_properties(actor_asset)
            dof_prop["driveMode"].fill(control_mode)
            self.gym.set_actor_dof_properties(env, humanoid_handle, dof_prop)
            
            #! create projectile actor
            self._build_proj(env, i)
            envs.append(env)

        #self.get_num_actors_per_env()
        self.num_envs = len(envs)


        return envs, humanoid_handles
    
    def _load_proj_asset(self):
        self._proj_handles = []
        asset_root = "./assets"
        small_asset_file = "cube.urdf"

        small_asset_options = gymapi.AssetOptions()
        small_asset_options.angular_damping = 0.01
        small_asset_options.linear_damping = 0.01
        small_asset_options.max_angular_velocity = 100.0
        small_asset_options.density = 20.0
        small_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        self._small_proj_asset = self.gym.load_asset(self.sim, asset_root, small_asset_file, small_asset_options)
        print("===== loaded proj asset =====")
        return
    

    def _build_proj(self, env_ptr, env_id):
        col_group = env_id
        col_filter = 0
        segmentation_id = 0

        for i, obj in enumerate(PERTURB_OBJS):
            default_pose = gymapi.Transform()
            default_pose.p.x = 200 + i
            default_pose.p.z = 1
            obj_type = obj[0]
            if (obj_type == "small"):
                proj_asset = self._small_proj_asset
            elif (obj_type == "large"):
                proj_asset = self._large_proj_asset

            proj_handle = self.gym.create_actor(env_ptr, proj_asset, default_pose, "proj{:d}".format(i), col_group, col_filter, segmentation_id)
            self._proj_handles.append(proj_handle)

        return
    
    def _get_num_objs(self):
        return len(PERTURB_OBJS)
    
    def _build_proj_tensors(self):
        num_actors = self.get_num_actors_per_env()  # n_humanoid + n_proj 
        num_objs = self._get_num_objs()

        self._proj_states = self._all_root_tensor.view(self.num_envs, num_actors, 13)[..., (num_actors - num_objs):, :]      # [n_envs, num_objs, 13]
        self._tar_actor_ids = torch.tensor(num_actors * np.arange(self.num_envs), device=self.device, dtype=torch.int32) + 1
        
        self._proj_actor_ids = num_actors * np.arange(self.num_envs)    # num_actors*[0, 1, 2, ... , n]
        self._proj_actor_ids = np.expand_dims(self._proj_actor_ids, axis=-1)        # num_actors*[[0], [1], [2] , ... , [n]]
                                # env의 첫 시작 index              # [1, 2, ..., n_projs] * n_envs
        self._proj_actor_ids = self._proj_actor_ids + np.reshape(np.array(self._proj_handles), [self.num_envs, num_objs])   # shape: (num_objs *num_envs, ) -> (self.num_envs, num_objs)
        self._proj_actor_ids = self._proj_actor_ids.flatten()
        self._proj_actor_ids = to_torch(self._proj_actor_ids, device=self.device, dtype=torch.int32)

        bodies_per_env = self._all_link_tensor.shape[0] // self.num_envs                # n_humanoid + n_proj  
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)               # [num_links + 1, 3]
        # 캐릭터 num_bodies 이후, 마지막 원소만 가져오기 [1, 3]    
        self._proj_contact_forces = contact_force_tensor.view(self.num_envs, bodies_per_env, 3)[..., (num_actors - num_objs):, :]

        return
    
    def create_tensors(self):
        super().create_tensors()
        self._build_proj_tensors()
        return 
    

    def _calc_perturb_times(self):
        self._perturb_timesteps = []
        total_steps = 0
        for i, obj in enumerate(PERTURB_OBJS):
            curr_time = obj[1]
            total_steps += curr_time
            self._perturb_timesteps.append(total_steps)
        self._perturb_timesteps = np.array(self._perturb_timesteps)
        return
        
    def _update_proj(self):

        curr_timestep = self.lifetime.cpu().numpy()[0]
        curr_timestep = curr_timestep % (self._perturb_timesteps[-1] + 1)
        perturb_step = np.where(self._perturb_timesteps == curr_timestep)[0]

        if (self.perterb):
            perturb_id = 0          # projectile 0 or 1
            n = self.num_envs
            humanoid_root_pos = self.root_tensor[..., 0:3]

            rand_theta = torch.rand([n], dtype=self._proj_states.dtype, device=self._proj_states.device)
            rand_theta *= 2 * np.pi
            rand_dist = (self._proj_dist_max - self._proj_dist_min) * torch.rand([n], dtype=self._proj_states.dtype, device=self._proj_states.device) + self._proj_dist_min
            # pos_x = rand_dist * torch.cos(rand_theta)
            # pos_y = -rand_dist * torch.sin(rand_theta)
            # pos_z = (self._proj_h_max - self._proj_h_min) * torch.rand([n], dtype=self._proj_states.dtype, device=self._proj_states.device) + self._proj_h_min
            
            tar_body_idx = self.tar_body_idx[0] if self.shoot == "right" else self.tar_body_idx[1]
            dir_body_idx = self.dir_body_idx[0] if self.shoot == "right" else self.dir_body_idx[1]
            heading = heading_zup(self.root_orient[0])
            UP_AXIS = 2
            up_dir = torch.zeros_like(self.root_pos[0])                                       # N x 1 x 3
            up_dir[..., UP_AXIS] = 1
            heading_orient = axang2quat(up_dir, heading) 
            offset = -0.7 if self.shoot == "right" else 0.7
            offset_vec = torch.tensor([0, offset, 0], dtype=torch.float32, device=self.device)
            ego_offset_vec = rotatepoint(heading_orient, offset_vec)
            
            launch_tar_pos = (self.link_pos[..., dir_body_idx, :] + self.link_pos[..., tar_body_idx, :])/2
            dir = launch_tar_pos[..., 0:3] - humanoid_root_pos[..., 0:3]
            dir = torch.nn.functional.normalize(dir, dim=-1)
            self._proj_states[..., perturb_id, 0] = humanoid_root_pos[..., 0] + 5 * ego_offset_vec[..., 0]
            self._proj_states[..., perturb_id, 1] = humanoid_root_pos[..., 1] + 5 * ego_offset_vec[..., 1]
            self._proj_states[..., perturb_id, 2] = humanoid_root_pos[..., 2] + dir[..., 2]
            
            # self._proj_states[..., perturb_id, 0] = self.spawn.x
            # self._proj_states[..., perturb_id, 1] = self.spawn.y
            # self._proj_states[..., perturb_id, 2] = self.spawn.z
            self._proj_states[..., perturb_id, 3:6] = 0.0
            self._proj_states[..., perturb_id, 6] = 1.0
            

            launch_dir = launch_tar_pos - self._proj_states[..., perturb_id, 0:3]
            
            launch_dir[..., 2] += 0.5 if self.shoot == "right" else 0.4
            #launch_dir += 0.1 * torch.randn_like(launch_dir)
            launch_dir = torch.nn.functional.normalize(launch_dir, dim=-1)
            launch_speed = (self._proj_speed_max - self._proj_speed_min) * torch.rand_like(launch_dir[:, 0:1]) + self._proj_speed_min
            launch_vel = launch_speed * launch_dir
            launch_vel[..., 0:2] += self.link_lin_vel[..., tar_body_idx, 0:2]
            
            self._proj_states[..., perturb_id, 7:10] = launch_vel
            self._proj_states[..., perturb_id, 10:13] = 0.0
            
            self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._all_root_tensor),
                                                        gymtorch.unwrap_tensor(self._proj_actor_ids),
                                                        len(self._proj_actor_ids))
            self.perterb = False
        return        
    
    def update_vobjects(self):
        self._update_proj()
        return

    def subscribe_keyboards_for_obj(self):
        # projectile
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_R, "right_shoot")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_T, "left_shoot")
        self.gym.subscribe_viewer_mouse_event(self.viewer, gymapi.MOUSE_LEFT_BUTTON, "mouse_shoot")
        return
    
    def update_obj_actions(self, event):
        # projectile
        if (event.action == "right_shoot" or event.action == "left_shoot") and event.value > 0:
            self.perterb = True
            self.shoot = "right" if event.action == "right_shoot" else "left"        

            # projectile shooting start position
            cam_pose = self.gym.get_viewer_camera_transform(self.viewer, self.envs[0])
            self.spawn = cam_pose.p
        
        return
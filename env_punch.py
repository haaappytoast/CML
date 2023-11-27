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

class ICCGANHumanoidStrike(ICCGANHumanoidVRControl):
    STRIKE_BODY_NAMES = ["right_hand", "right_lower_arm"]

    def __init__(self, *args, 
                sensor_inputs: Optional[Dict[str, SensorInputConfig]]=None,
                 **kwargs):
        self.sensor_inputs = sensor_inputs

        self._tar_speed = 1.0
        self._tar_dist_max = 0.5
        self._tar_height_min = 1.0
        self._tar_height_max = 1.5
        self.time = 0
        self.reach = False
        self.resett = False

        self.strike_body_names = parse_kwarg(kwargs, "strike_body_names", self.STRIKE_BODY_NAMES)

        super().__init__(*args, sensor_inputs = self.sensor_inputs, **kwargs)
        
        self._tar_pos = -100 * torch.ones([self.num_envs, 3], device=self.device, dtype=torch.float)
        
        reach_body = ["right_hand", "left_hand"]

        self._reach_body_idx = []
        
        for i in range(len(reach_body)): 
            self._reach_body_idx.append(self._build_reach_body_id_tensor(self.envs[0], self.actors[0], reach_body[i]))

    def create_envs(self, n: int):
        #! location_marker assets 
        self._marker_handles = []
        self._load_obj_asset()

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
            
            #! create location_marker actor
            self._build_obj(env, i)
            envs.append(env)

        #self.get_num_actors_per_env()
        self.num_envs = len(envs)


        return envs, humanoid_handles
    
    def create_tensors(self):
        super().create_tensors()
        self._build_obj_state_tensors()
        return 
        
    def _load_obj_asset(self):
        self._target_handles = []
        asset_root = "./assets"
        asset_file = "strike_target.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.linear_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.density = 10.0
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        self._target_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        print("===== loaded target asset =====")
        return
    
    def _build_obj(self, env_ptr, env_id):
        col_group = env_id
        col_filter = 0
        segmentation_id = 0

        default_pose = gymapi.Transform()
        default_pose.p.x = -0.3
        default_pose.p.y = 3.2
        default_pose.p.z = 0.9
        
        target_handle = self.gym.create_actor(env_ptr, self._target_asset, default_pose, "target", col_group, col_filter, segmentation_id)
        self._target_handles.append(target_handle)
        return

    def _build_obj_state_tensors(self):
        num_actors = self.get_num_actors_per_env()  # n_humanoid + 1 
        self._marker_states = self._all_root_tensor.view(self.num_envs, num_actors, self._all_root_tensor.shape[-1])[..., num_actors-1:, :]
        self._marker_pos = self._marker_states[..., :3]

        self._marker_actor_ids = self._humanoid_actor_ids + 1

        # marker contact force
        self._tar_contact_forces = self.contact_force_tensor[..., self.num_bodies:, :]
        return
    
    def _build_reach_body_id_tensor(self, env_ptr, actor_handle, body_name):
        body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name)
        assert(body_id != -1)
        body_id = to_torch(body_id, device=self.device, dtype=torch.long)
        return body_id
    
    def _update_obj(self):
        if self.reach:
            self._update_task()
            self._update_target()
            self.reach = False
        if self.resett:
            self._tar_pos[..., 0:3] = torch.tensor([-100, -100, -100])
            self._update_target()
            self.resett = False
        return    

    def map_xyoffset(self):
        if self.time == 0:
            xoffset, yoffset = -0.2, 3
        if self.time == 1:
            xoffset, yoffset = -0.75, 3.55
        else:
            xoffset, yoffset = 0.4, 6.0

        return xoffset, yoffset
    def _update_task(self):
        env_ids = self.envs
        n = len(env_ids)
        heading = heading_zup(self.root_orient[0])
        UP_AXIS = 2
        up_dir = torch.zeros_like(self.root_pos[0])                                       # N x 1 x 3
        up_dir[..., UP_AXIS] = 1
        heading_orient = axang2quat(up_dir, heading) 
        
        xoffset, yoffset = self.map_xyoffset()

        offset_vec = torch.tensor([xoffset, yoffset, 0], dtype=torch.float32, device=self.device)
        ego_offset_vec = rotatepoint(heading_orient, offset_vec)        

        print(" tar_pos changed!")
        self._tar_pos[..., 0:3] = self.root_pos + offset_vec
        self._tar_pos[..., 2] = 0.95
        return

    def _update_target(self):
        self._marker_pos[..., :] = self._tar_pos
        self._marker_states[..., 3:6] = 0.0                 # quaternion
        self._marker_states[..., 6] = 1.0                   # quaternion
        self._marker_states[..., 7:13] = 0.0                # lin_vel, ang_vel
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._all_root_tensor),
                                                    gymtorch.unwrap_tensor(self._marker_actor_ids), len(self._marker_actor_ids))
        
        return

    def _draw_task(self):
            
        cols = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
        self.gym.clear_lines(self.viewer)
        reach_idx = self._reach_body_idx[0]
        starts = self.link_pos[:, reach_idx, :]
        ends = self._tar_pos

        verts = torch.cat([starts, ends], dim=-1).cpu().numpy()

        for i, env_ptr in enumerate(self.envs):
            curr_verts = verts[i]
            curr_verts = curr_verts.reshape([1, 6])
            self.gym.add_lines(self.viewer, env_ptr, curr_verts.shape[0], curr_verts, cols)

        return
    
    def update_vobjects(self):
        self._update_obj()
        return

    def subscribe_keyboards_for_obj(self):
        # place object near hands
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_1, "place_target")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_2, "reset")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_3, "temp")
        return
    
    def update_obj_actions(self, event):
        # projectile
        if (event.action == "place_target") and event.value > 0:
            self.reach = True
            self.time+=1
            print("place target!")
        if (event.action == "reset") and event.value > 0:
            self.resett = True
            print("reset!")
        if (event.action == "temp") and event.value > 0:
            print("self.root_pos: ", self.root_pos)
        return

    def update_viewer(self):
        self._draw_task()
        super().update_viewer()
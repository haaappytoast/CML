from typing import Callable, Optional, List, Dict, Any
from collections import namedtuple
import os
from isaacgym import gymapi, gymtorch
import torch
import utils
from utils import heading_zup, axang2quat, rotatepoint, quatconj, quatmultiply, quatdiff_normalized, quat_inverse
from poselib.core import quat_mul
from isaacgym import gymutil
from humanoid_view import HumanoidView
from humanoid_extract import HumanoidExtract, ICCGANHumanoidExtractTarget

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
              xy_pressed: Optional[str]=None):
        
        self.rlh_localPos = rlh_localPos
        self.rlh_localRot = rlh_localRot
        self.xy_pressed = xy_pressed
        
SensorInputProperty = namedtuple("SensorInputProperty",
    "name rlh_localPos rlh_localRot xy_pressed")

class Env(object):
    UP_AXIS = 2
    CHARACTER_MODEL = None

    def __init__(self,
        n_envs: int, fps: int, frameskip: int,
        episode_length: Optional[Callable or int] = 300,
        control_mode: str = "position",
        compute_device: int = 0,
        graphics_device: Optional[int] = None,
        character_model: Optional[str] = None,
        **kwargs
    ):
        self.viewer = None
        assert(control_mode in ["position", "torque", "free"])
        self.frameskip = frameskip
        self.fps = fps
        self.step_time = 1./self.fps
        self.control_mode = control_mode
        self.episode_length = episode_length
        self.device = torch.device(compute_device)
        if graphics_device is None:
            graphics_device = compute_device
        self.character_model = self.CHARACTER_MODEL if character_model is None else character_model

        sim_params = self.setup_sim_params()
        self.gym = gymapi.acquire_gym()
        self.sim = self.gym.create_sim(compute_device, graphics_device, gymapi.SIM_PHYSX, sim_params)
        self.add_ground()
        self.envs, self.actors = self.create_envs(n_envs)
        self.setup_action_normalizer()
        self.create_tensors()

        self.gym.prepare_sim(self.sim)

        n_actors_per_env = self.gym.get_actor_count(self.envs[0])
        self.actor_ids = torch.arange(n_actors_per_env * len(self.envs), dtype=torch.int32, device=self.device).view(len(self.envs), -1)
        controllable_actors = []
        for i in range(self.gym.get_actor_count(self.envs[0])):
            dof = self.gym.get_actor_dof_count(self.envs[0], i)
            if dof > 0: controllable_actors.append(i)
        self.actor_ids_having_dofs = \
            n_actors_per_env * torch.arange(len(self.envs), dtype=torch.int32, device=self.device).unsqueeze(-1) + \
            torch.tensor(controllable_actors, dtype=torch.int32, device=self.device).unsqueeze(-2)
        
        self.refresh_tensors()
        self.viewer_pause = False
        self.camera_following = True
        self.viewer_advance = False
        tar_env = len(self.envs)//4 + int(len(self.envs)**0.5)//2
        base_pos = self.root_pos[tar_env].cpu()
        self.cam_target = gymapi.Vec3(*self.vector_up(1.0, [base_pos[0], base_pos[1], base_pos[2]]))

        self.simulation_step = 0
        self.lifetime = torch.zeros(len(self.envs), dtype=torch.int64, device=self.device)
        self.done = torch.ones(len(self.envs), dtype=torch.bool, device=self.device)
        self.info = dict(lifetime=self.lifetime)

        self.act_dim = self.action_scale.size(-1)
        self.ob_dim = self.observe().size(-1)
        self.rew_dim = self.reward().size(-1)

        for i in range(self.gym.get_actor_count(self.envs[0])):
            rigid_body = self.gym.get_actor_rigid_body_dict(self.envs[0], i)
            print(rigid_body, len(rigid_body))
            dof = self.gym.get_actor_dof_dict(self.envs[0], i)
            print(dof, len(dof))

    def __del__(self):
        if hasattr(self, "viewer") and self.viewer is not None:
            self.gym.destroy_viewer(self.viewer)
        if hasattr(self, "sim"):
            self.gym.destroy_sim(self.sim)

    def vector_up(self, val: float, base_vector=None):
        if base_vector is None:
            base_vector = [0., 0., 0.]
        base_vector[self.UP_AXIS] = val
        return base_vector
    
    def setup_sim_params(self):
        sim_params = gymapi.SimParams()
        sim_params.use_gpu_pipeline = True # force to enable GPU
        sim_params.dt = self.step_time/self.frameskip
        sim_params.substeps = 2
        sim_params.up_axis = gymapi.UP_AXIS_Z if self.UP_AXIS == 2 else gymapi.UP_AXIS_Y
        sim_params.gravity = gymapi.Vec3(*self.vector_up(-9.81))
        sim_params.num_client_threads = 0
        sim_params.physx.num_threads = 4
        sim_params.physx.solver_type = 1
        sim_params.physx.num_subscenes = 4  # works only for CPU 
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 0
        sim_params.physx.contact_offset = 0.01
        sim_params.physx.rest_offset = 0.0
        sim_params.physx.bounce_threshold_velocity = 0.2
        sim_params.physx.max_depenetration_velocity = 10.0
        sim_params.physx.default_buffer_size_multiplier = 5.0
        sim_params.physx.max_gpu_contact_pairs = 8*1024*1024
        # FIXME IsaacGym Pr4 will provide unreliable results when collecting from all substeps
        sim_params.physx.contact_collection = \
            gymapi.ContactCollection(gymapi.ContactCollection.CC_LAST_SUBSTEP) 
        #gymapi.ContactCollection(gymapi.ContactCollection.CC_ALL_SUBSTEPS)
        sim_params.physx.use_gpu = True
        return sim_params

    def add_ground(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(*self.vector_up(1.0))
        plane_params.static_friction = 1.0
        plane_params.dynamic_friction = 1.0
        plane_params.restitution = 0.0
        self.gym.add_ground(self.sim, plane_params)

    def add_actor(self, env, i):
        pass

    def create_envs(self, n: int):
        if self.control_mode == "position":
            control_mode = gymapi.DOF_MODE_POS
        elif self.control_mode == "torque":
            control_mode = gymapi.DOF_MODE_EFFORT
        else:
            control_mode = gymapi.DOF_MODE_NONE

        envs, actors = [], []
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
        for i in range(n):
            env = self.gym.create_env(self.sim, spacing_lower, spacing_upper, n_envs_per_row)
            actor = self.gym.create_actor(env, actor_asset, start_pose, "actor", i, -1, 0)
            self.add_actor(env, i)
            envs.append(env)
            actors.append(actor)
            # enable PD control
            # Kp (stiffness) and Kd (damping) are defined inside the mjcf xml file
            dof_prop = self.gym.get_asset_dof_properties(actor_asset)
            dof_prop["driveMode"].fill(control_mode)
            self.gym.set_actor_dof_properties(env, actor, dof_prop)
        return envs, actors

    def render(self):
        tar_env = len(self.envs)//4 + int(len(self.envs)**0.5)//2
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        base_pos = self.root_pos[tar_env].cpu()
        cam_pos = gymapi.Vec3(*self.vector_up(2.0, [base_pos[0], base_pos[1]-4.5, base_pos[2]-4.5]))
        self.gym.viewer_camera_look_at(self.viewer, self.envs[tar_env], cam_pos, self.cam_target)
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_ESCAPE, "QUIT")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_F, "TOGGLE_CAMERA_FOLLOWING")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_P, "TOGGLE_PAUSE")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_D, "SINGLE_STEP_ADVANCE")
    
    def update_viewer(self):
        self.gym.poll_viewer_events(self.viewer)
        for event in self.gym.query_viewer_action_events(self.viewer):
            if event.action == "QUIT" and event.value > 0:
                exit()
            if event.action == "TOGGLE_CAMERA_FOLLOWING" and event.value > 0:
                self.camera_following = not self.camera_following
            if event.action == "TOGGLE_PAUSE" and event.value > 0:
                self.viewer_pause = not self.viewer_pause
            if event.action == "SINGLE_STEP_ADVANCE" and event.value > 0:
                self.viewer_advance = not self.viewer_advance
        if self.camera_following: self.update_camera()
        self.gym.step_graphics(self.sim)

    def update_camera(self):
        tar_env = len(self.envs)//4 + int(len(self.envs)**0.5)//2
        cam_trans = self.gym.get_viewer_camera_transform(self.viewer, self.envs[tar_env])
        dx, dy = cam_trans.p.x - self.cam_target.x, cam_trans.p.y - self.cam_target.y
        base_pos = self.root_pos[tar_env].cpu()
        cam_pos = gymapi.Vec3(base_pos[0]+dx, base_pos[1]+dy, cam_trans.p.z)
        self.cam_target = gymapi.Vec3(base_pos[0], base_pos[1], 1.0)
        self.gym.viewer_camera_look_at(self.viewer, self.envs[tar_env], cam_pos, self.cam_target)

    def refresh_tensors(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

    def create_tensors(self):
        root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        root_tensor = gymtorch.wrap_tensor(root_tensor)
        self.root_tensor = root_tensor.view(len(self.envs), -1, 13)

        num_links = self.gym.get_env_rigid_body_count(self.envs[0])
        link_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        link_tensor = gymtorch.wrap_tensor(link_tensor)
        self.link_tensor = link_tensor.view(len(self.envs), num_links, -1)

        num_dof = self.gym.get_env_dof_count(self.envs[0])
        joint_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        joint_tensor = gymtorch.wrap_tensor(joint_tensor)
        # IsaacGym Preview 3 supports fix, revolute and prismatic (1d) joints only
        self.joint_tensor = joint_tensor.view(len(self.envs), num_dof, -1)  # n_envs x n_dof x 2 (pos, vel for each)

        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)
        self.contact_force_tensor = contact_force_tensor.view(len(self.envs), -1, 3)

    def setup_action_normalizer(self):
        action_lower, action_upper = [], []
        action_scale = []
        for i in range(self.gym.get_actor_count(self.envs[0])):
            actor = self.gym.get_actor_handle(self.envs[0], i)
            n_dof = self.gym.get_actor_dof_count(self.envs[0], actor)
            if n_dof < 1: continue
            dof_prop = self.gym.get_actor_dof_properties(self.envs[0], actor)
            if self.control_mode == "position":
                action_lower.extend([min(dof_prop["lower"][j], dof_prop["upper"][j]) for j in range(n_dof)])
                action_upper.extend([max(dof_prop["lower"][j], dof_prop["upper"][j]) for j in range(n_dof)])
                action_scale.append(2)
            elif self.control_mode == "torque":
                action_lower.extend([-dof_prop["effort"][j] for j in range(n_dof)])
                action_upper.extend([dof_prop["effort"][j] for j in range(n_dof)])
                action_scale.append(1)
        action_offset = 0.5 * np.add(action_upper, action_lower)
        action_scale *= 0.5 * np.subtract(action_upper, action_lower)
        self.action_offset = torch.tensor(action_offset, dtype=torch.float32, device=self.device)
        self.action_scale = torch.tensor(action_scale, dtype=torch.float32, device=self.device)

    def process_actions(self, actions):
        if type(actions) is tuple and len(actions) is 1:
            actions = actions[0]
            
        return actions*self.action_scale + self.action_offset

    def reset(self):
        self.lifetime.zero_()
        self.done.fill_(True)
        self.info = dict(lifetime=self.lifetime)
        self.request_quit = False
        self.obs = None

    def reset_done(self):
        if not self.viewer_pause:
            env_ids = torch.nonzero(self.done).view(-1)
            if len(env_ids):
                self.reset_envs(env_ids)                                # 1. reset_envs를 통해 goal도 reset한다!
                if len(env_ids) == len(self.envs) or self.obs is None:
                    self.obs = self.observe()                           # 2. 이후 새로운 goal과 state를 가지고 다음 obs observe!
                else:
                    self.obs[env_ids] = self.observe(env_ids)
        return self.obs, self.info
    
    def reset_envs(self, env_ids):
        ref_root_tensor, ref_link_tensor, ref_joint_tensor = self.init_state(env_ids)

        self.root_tensor[env_ids] = ref_root_tensor
        self.link_tensor[env_ids] = ref_link_tensor
        self.joint_tensor[env_ids] = ref_joint_tensor

        actor_ids = self.actor_ids[env_ids].flatten()
        n_actor_ids = len(actor_ids)
        actor_ids = gymtorch.unwrap_tensor(actor_ids)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
            gymtorch.unwrap_tensor(self.root_tensor),
            actor_ids, n_actor_ids
        )
        actor_ids = self.actor_ids_having_dofs[env_ids].flatten()
        n_actor_ids = len(actor_ids)
        actor_ids = gymtorch.unwrap_tensor(actor_ids)
        self.gym.set_dof_state_tensor_indexed(self.sim,
            gymtorch.unwrap_tensor(self.joint_tensor),
            actor_ids, n_actor_ids
        )

        self.lifetime[env_ids] = 0

    def do_simulation(self):
        for _ in range(self.frameskip):
            self.gym.simulate(self.sim)
        self.simulation_step += 1

    def step(self, actions):
        if not self.viewer_pause or self.viewer_advance:
            self.apply_actions(actions)
            self.do_simulation()
            self.refresh_tensors()
            self.lifetime += 1
            if self.viewer is not None:
                self.gym.fetch_results(self.sim, True)
                self.viewer_advance = False

        if self.viewer is not None:
            self.update_viewer()
            self.gym.draw_viewer(self.viewer, self.sim, True)
            self.gym.sync_frame_time(self.sim)    # sync to simulation dt

        rewards = self.reward()
        terminate = self.termination_check()                    # N
        if self.viewer_pause:
            overtime = None
        else:
            overtime = self.overtime_check()
        if torch.is_tensor(overtime):
            self.done = torch.logical_or(overtime, terminate)
        else:
            self.done = terminate
        self.info["terminate"] = terminate
        self.obs = self.observe()
        self.request_quit = False if self.viewer is None else self.gym.query_viewer_has_closed(self.viewer)
        return self.obs, rewards, self.done, self.info

    def apply_actions(self, actions):
        actions = self.process_actions(actions)
        actions = gymtorch.unwrap_tensor(actions)
        if self.control_mode == "position":
            self.gym.set_dof_position_target_tensor(self.sim, actions)
        elif self.control_mode == "torque":
            self.gym.set_dof_actuation_force_tensor(self.sim, actions)
        else:
            self.gym.set_dof_state_tensor(self.sim, actions)

    def init_state(self, env_ids):
        pass
    
    def observe(self, env_ids=None):
        pass
    
    def overtime_check(self):
        if self.episode_length:
            if callable(self.episode_length):
                return self.lifetime >= self.episode_length(self.simulation_step)
            return self.lifetime >= self.episode_length
        return None

    def termination_check(self):
        return torch.zeros(len(self.envs), dtype=torch.bool, device=self.device)

    def reward(self):
        return torch.ones((len(self.envs), 0), dtype=torch.float32, device=self.device)

    def set_char_color(self, col, env_ids, key_links=None):
        n_links = self.char_link_tensor.size(1) 
        for env_id in env_ids:
            env_ptr = self.envs[env_id]
            handle = self.actors[env_id]

        if len(env_ids):
            if key_links is None:
                for j in range(n_links):
                    self.gym.set_rigid_body_color(env_ptr, handle, j, gymapi.MESH_VISUAL,
                                                gymapi.Vec3(col[0], col[1], col[2]))
            else:
                for j in key_links:
                    self.gym.set_rigid_body_color(env_ptr, handle, j, gymapi.MESH_VISUAL,
                                                gymapi.Vec3(col[0], col[1], col[2]))
        return


from ref_motion import ReferenceMotion
import numpy as np


class ICCGANHumanoid(Env):

    CHARACTER_MODEL = "assets/humanoid.xml"
    CONTROLLABLE_LINKS = ["torso", "head", 
        "right_upper_arm", "right_lower_arm",
        "left_upper_arm", "left_lower_arm", 
        "right_thigh", "right_shin", "right_foot",
        "left_thigh", "left_shin", "left_foot"]
    DOFS =  [3, 3, 3, 1, 3, 1, 3, 1, 3, 3, 1, 3]
    DOF_OFFSET =  [0, 0, 3, 6, 9, 9, 10, 13, 13, 14, 17, 18, 21, 24, 25, 28]
    CONTACTABLE_LINKS = ["right_foot", "left_foot"]
    UP_AXIS = 2

    GOAL_DIM = 0
    GOAL_REWARD_WEIGHT = None
    ENABLE_GOAL_TIMER = False
    GOAL_TENSOR_DIM = None

    OB_HORIZON = 4

    def __init__(self, *args,
        motion_file: str,
        discriminators: Dict[str, DiscriminatorConfig],
    **kwargs):
        contactable_links = parse_kwarg(kwargs, "contactable_links", self.CONTACTABLE_LINKS)
        controllable_links = parse_kwarg(kwargs, "controllable_links", self.CONTROLLABLE_LINKS)
        dofs = parse_kwarg(kwargs, "dofs", self.DOFS)
        goal_reward_weight = parse_kwarg(kwargs, "goal_reward_weight", self.GOAL_REWARD_WEIGHT)
        self.enable_goal_timer = parse_kwarg(kwargs, "enable_goal_timer", self.ENABLE_GOAL_TIMER)
        self.goal_tensor_dim = parse_kwarg(kwargs, "goal_tensor_dim", self.GOAL_TENSOR_DIM)
        self.ob_horizon = parse_kwarg(kwargs, "ob_horizon", self.OB_HORIZON)
        super().__init__(*args, **kwargs)

        n_envs = len(self.envs)
        n_links = self.char_link_tensor.size(1)
        n_dofs = self.char_joint_tensor.size(1)
        
        controllable_links = [self.gym.find_actor_rigid_body_handle(self.envs[0], self.actors[0], link)
            for link in controllable_links]

        if contactable_links:
            contact = np.zeros((n_envs, n_links), dtype=bool)
            for link in contactable_links:
                lid = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actors[0], link)
                assert(lid >= 0), "Unrecognized contactable link {}".format(link)
                contact[:, lid] = True
            self.contactable_links = torch.tensor(contact).to(self.contact_force_tensor.device)
        else:
            self.contactable_links = None

        init_pose = motion_file
        self.ref_motion = ReferenceMotion(motion_file=init_pose, character_model=self.character_model,
            key_links=np.arange(n_links), controllable_links=controllable_links, dofs=dofs,
            device=self.device
        )

        ref_motion = {init_pose: self.ref_motion}
        disc_ref_motion = dict()
        for id, config in discriminators.items():
            m = init_pose if config.motion_file is None else config.motion_file
            if m not in ref_motion:
                ref_motion[m] = ReferenceMotion(motion_file=m, character_model=self.character_model,
                    key_links=np.arange(n_links), controllable_links=controllable_links, dofs=dofs,
                    device=self.device
                )
            key = (ref_motion[m], config.replay_speed)
            if config.ob_horizon is None:
                config.ob_horizon = self.ob_horizon+1
            if key not in disc_ref_motion: disc_ref_motion[key] = [0, []]
            disc_ref_motion[key][0] = max(disc_ref_motion[key][0], config.ob_horizon)
            disc_ref_motion[key][1].append(id)

        if goal_reward_weight is not None:
            reward_weights = torch.empty((len(self.envs), self.rew_dim), dtype=torch.float32, device=self.device)
            if not hasattr(goal_reward_weight, "__len__"):
                goal_reward_weight = [goal_reward_weight]
            assert(self.rew_dim == len(goal_reward_weight))
            for i, w in zip(range(self.rew_dim), goal_reward_weight):
                reward_weights[:, i] = w
        elif self.rew_dim:
            goal_reward_weight = []
            assert(self.rew_dim == len(goal_reward_weight))

        n_comp = len(discriminators) + self.rew_dim
        if n_comp > 1:
            self.reward_weights = torch.zeros((n_envs, n_comp), dtype=torch.float32, device=self.device)
            weights = [disc.weight for _, disc in discriminators.items() if disc.weight is not None]
            total_weights = sum(weights) if weights else 0
            assert(total_weights <= 1), "Discriminator weights must not be greater than 1."
            n_unassigned = len(discriminators) - len(weights)
            rem = 1 - total_weights
            for disc in discriminators.values():
                if disc.weight is None:
                    disc.weight = rem / n_unassigned
                elif n_unassigned == 0:
                    disc.weight /= total_weights
        else:
            self.reward_weights = None

        self.discriminators = dict()
        max_ob_horizon = self.ob_horizon+1
        for i, (id, config) in enumerate(discriminators.items()):
            key_links = None if config.key_links is None else sorted([
                self.gym.find_actor_rigid_body_handle(self.envs[0], self.actors[0], link) for link in config.key_links
            ])
            parent_link = None if config.parent_link is None else \
                self.gym.find_actor_rigid_body_handle(self.envs[0], self.actors[0], config.parent_link)

            assert(key_links is None or all(lid >= 0 for lid in key_links))
            assert(parent_link is None or parent_link >= 0)
            
            self.discriminators[id] = DiscriminatorProperty(
                name = id,
                key_links = key_links,
                parent_link = parent_link,
                local_pos = config.local_pos,
                local_height = config.local_height,
                replay_speed = config.replay_speed,
                ob_horizon = config.ob_horizon,
                id=i
            )
            if self.reward_weights is not None:
                self.reward_weights[:, i] = config.weight
            max_ob_horizon = max(max_ob_horizon, config.ob_horizon)

        if max_ob_horizon != self.state_hist.size(0):
            self.state_hist = torch.empty((max_ob_horizon, *self.state_hist.shape[1:]),
                dtype=self.root_tensor.dtype, device=self.device)
        self.disc_ref_motion = [
            (ref_motion, replay_speed, max_ob_horizon, [self.discriminators[id] for id in disc_ids])
            for (ref_motion, replay_speed), (max_ob_horizon, disc_ids) in disc_ref_motion.items()
        ]

        if self.rew_dim > 0:
            if self.rew_dim > 1:
                self.reward_weights *= (1-reward_weights.sum(dim=-1, keepdim=True))
            else:
                self.reward_weights *= (1-reward_weights)
            self.reward_weights[:, -self.rew_dim:] = reward_weights
            
        self.info["ob_seq_lens"] = torch.zeros_like(self.lifetime)  # dummy result
        self.info["disc_obs"] = self.observe_disc(self.state_hist)  # dummy result
        self.info["disc_obs_expert"] = self.info["disc_obs"]        # dummy result
        self.goal_dim = self.GOAL_DIM
        self.state_dim = (self.ob_dim-self.goal_dim)//self.ob_horizon
        self.disc_dim = {
            name: ob.size(-1)
            for name, ob in self.info["disc_obs"].items()
        }
    
    def reset_done(self):
        obs, info = super().reset_done()
        info["ob_seq_lens"] = self.ob_seq_lens
        info["reward_weights"] = self.reward_weights
        return obs, info
    
    def reset(self):
        if self.goal_tensor is not None:
            self.goal_tensor.zero_()
            if self.goal_timer is not None: self.goal_timer.zero_()
        super().reset()

    def reset_envs(self, env_ids):
        super().reset_envs(env_ids)
        self.reset_goal(env_ids)
        
    def reset_goal(self, env_ids):
        pass
    
    def step(self, actions):
        self.state_hist[:-1] = self.state_hist[1:].clone()
        obs, rews, dones, info = super().step(actions)
        info["disc_obs"] = self.observe_disc(self.state_hist)
        info["disc_obs_expert"], info["disc_seq_len"] = self.fetch_real_samples()
        return obs, rews, dones, info

    def overtime_check(self):
        if self.goal_timer is not None:
            self.goal_timer -= 1
            env_ids = torch.nonzero(self.goal_timer <= 0).view(-1)
            if len(env_ids) > 0: self.reset_goal(env_ids)
        return super().overtime_check()

    def termination_check(self):
        if self.contactable_links is None:
            return torch.zeros_like(self.done)
        masked_contact = self.char_contact_force_tensor.clone()
        masked_contact[self.contactable_links] = 0          # N x n_links x 3

        contacted = torch.any(masked_contact > 1., dim=-1)  # N x n_links
        too_low = self.link_pos[..., self.UP_AXIS] < 0.15    # N x n_links

        terminate = torch.any(torch.logical_and(contacted, too_low), -1)    # N x
        terminate *= (self.lifetime > 1)
        return terminate

    def init_state(self, env_ids):
        motion_ids, motion_times = self.ref_motion.sample(len(env_ids))
        return self.ref_motion.state(motion_ids, motion_times)
    
    def create_tensors(self):
        super().create_tensors()
        n_dofs = self.gym.get_actor_dof_count(self.envs[0], 0)
        n_links = self.gym.get_actor_rigid_body_count(self.envs[0], 0)
        self.root_pos, self.root_orient = self.root_tensor[:, 0, :3], self.root_tensor[:, 0, 3:7]
        self.root_lin_vel, self.root_ang_vel = self.root_tensor[:, 0, 7:10], self.root_tensor[:, 0, 10:13]
        self.char_root_tensor = self.root_tensor[:, 0]
        if self.link_tensor.size(1) > n_links:
            self.link_pos, self.link_orient = self.link_tensor[:, :n_links, :3], self.link_tensor[:, :n_links, 3:7]
            self.link_lin_vel, self.link_ang_vel = self.link_tensor[:, :n_links, 7:10], self.link_tensor[:, :n_links, 10:13]
            self.char_link_tensor = self.link_tensor[:, :n_links]
        else:
            self.link_pos, self.link_orient = self.link_tensor[..., :3], self.link_tensor[..., 3:7]
            self.link_lin_vel, self.link_ang_vel = self.link_tensor[..., 7:10], self.link_tensor[..., 10:13]
            self.char_link_tensor = self.link_tensor
        if self.joint_tensor.size(1) > n_dofs:
            self.joint_pos, self.joint_vel = self.joint_tensor[:, :n_dofs, 0], self.joint_tensor[:, :n_dofs, 1]
            self.char_joint_tensor = self.joint_tensor[:, :n_dofs]
        else:
            self.joint_pos, self.joint_vel = self.joint_tensor[..., 0], self.joint_tensor[..., 1]
            self.char_joint_tensor = self.joint_tensor
        
        self.char_contact_force_tensor = self.contact_force_tensor[:, :n_links]

        ob_disc_dim = 13 + n_links*13
        self.state_hist = torch.empty((self.ob_horizon+1, len(self.envs), ob_disc_dim),
            dtype=self.root_tensor.dtype, device=self.device)                           # [5, NUM_ENVS, 13 + n_links*13]

        if self.goal_tensor_dim:
            try:
                self.goal_tensor = [
                    torch.zeros((len(self.envs), dim), dtype=self.root_tensor.dtype, device=self.device)
                    for dim in self.goal_tensor_dim
                ]
            except TypeError:
                self.goal_tensor = torch.zeros((len(self.envs), self.goal_tensor_dim), dtype=self.root_tensor.dtype, device=self.device)
        else:
            self.goal_tensor = None
        self.goal_timer = torch.zeros((len(self.envs), ), dtype=torch.int32, device=self.device) if self.enable_goal_timer else None

    def observe(self, env_ids=None):
        self.ob_seq_lens = self.lifetime+1 #(self.lifetime+1).clip(max=self.state_hist.size(0)-1)
        n_envs = len(self.envs)
        if env_ids is None or len(env_ids) == n_envs:
            self.state_hist[-1] = torch.cat((
                self.char_root_tensor, self.char_link_tensor.view(n_envs, -1)
            ), -1)
            env_ids = None
        else:
            n_envs = len(env_ids)
            self.state_hist[-1, env_ids] = torch.cat((
                self.char_root_tensor[env_ids], self.char_link_tensor[env_ids].view(n_envs, -1)
            ), -1)
        return self._observe(env_ids)
    
    def _observe(self, env_ids):
        if env_ids is None:
            return observe_iccgan(
                self.state_hist[-self.ob_horizon:], self.ob_seq_lens
            )
        else:
            return observe_iccgan(
                self.state_hist[-self.ob_horizon:][:, env_ids], self.ob_seq_lens[env_ids]
            )
    
    def observe_disc(self, state):
        seq_len = self.info["ob_seq_lens"]+1
        res = dict()
        if torch.is_tensor(state):
            # fake
            for id, disc in self.discriminators.items():
                res[id] = observe_disc(state[-disc.ob_horizon:], seq_len, disc.key_links, disc.parent_link, disc.local_pos, disc.local_height)
            return res
        else:
            # real
            seq_len_ = dict()
            for disc_name, s in state.items():
                disc = self.discriminators[disc_name]
                res[disc_name] = observe_disc(s[-disc.ob_horizon:], seq_len, disc.key_links, disc.parent_link, disc.local_pos, disc.local_height)
                seq_len_[disc_name] = seq_len
            return res, seq_len_

    def fetch_real_samples(self):
        n_inst = len(self.envs)

        samples = dict()
        for ref_motion, replay_speed, ob_horizon, discs in self.disc_ref_motion:
            dt = self.step_time
            if replay_speed is not None:
                dt /= replay_speed(n_inst)
            motion_ids, motion_times0 = ref_motion.sample(n_inst, truncate_time=dt*(ob_horizon-1))
            motion_ids = np.tile(motion_ids, ob_horizon)
            motion_times = np.concatenate((motion_times0, *[motion_times0+dt*i for i in range(1, ob_horizon)]))
            root_tensor, link_tensor, joint_tensor = ref_motion.state(motion_ids, motion_times)
            real = torch.cat((
                root_tensor, link_tensor.view(root_tensor.size(0), -1)
            ), -1).view(ob_horizon, n_inst, -1)

            for d in discs: samples[d.name] = real
        return self.observe_disc(samples)

    # def update_viewer(self):
    #     super().update_viewer()
    #     self.gym.clear_lines(self.viewer)
    #     self.visualize_axis(self.link_pos[:, [0, 6, 7, 8], :], self.link_orient[:, [0, 6, 7, 8], :], scale = 0.2, y=False, z =False)

    def visualize_axis(self, gpos, gquat, scale, x=True, y=True, z = True):
        gquat = gquat.view(-1, 4).cpu()                                                 # [num_envs x n_links, 4]
        tan_norm = utils.quat_to_tan_norm(gquat).cpu()
        rot_mat = utils.tan_norm_to_rotmat(tan_norm).cpu()
        tan, binorm, norm = rot_mat[..., 0:3], rot_mat[..., 3:6], rot_mat[..., 6:]       # [num_envs x n_links, 3]
        tan, binorm, norm = tan.view(len(self.envs), -1, 3), norm.view(len(self.envs), -1, 3), binorm.view(len(self.envs), -1, 3)   # [num_envs, n_links, 3]
        
        start = gpos.cpu().numpy()                                                      # [5,n_links,3]

        x_end = (gpos.cpu() + tan * scale).cpu().numpy()
        z_end = (gpos.cpu() + binorm * scale).cpu().numpy()
        y_end = (gpos.cpu() + norm * scale).cpu().numpy()
        
        n_lines = 5

        # x-axis
        if x:
            for j in range(gpos.size(1)):
                x_lines = np.stack([
                    np.stack((start[:, j, 0], start[:, j, 1], start[:, j, 2]+0.0015*i, x_end[:, j, 0], x_end[:, j, 1], x_end[:, j, 2]+0.0015*i), -1)
                            for i in range(n_lines)], -2)                                      # [n_envs, n_lines, 6]
                for e, l in zip(self.envs, x_lines):
                    self.gym.add_lines(self.viewer, e, n_lines, l, [[1., 0., 0.] for _ in range(n_lines)])
        # y-axis
        if y:
            for j in range(gpos.size(1)):
                ylines = np.stack([
                    np.stack((start[:, j, 0]+0.0015*i, start[:, j, 1], start[:, j, 2], y_end[:, j, 0]+0.0015*i, y_end[:, j, 1], y_end[:, j, 2]), -1)
                            for i in range(n_lines)], -2)                                      # [n_envs, n_lines, 6]
                for e, l in zip(self.envs, ylines):
                    self.gym.add_lines(self.viewer, e, n_lines, l, [[0., 1., 0.] for _ in range(n_lines)])
        # z-axis
        if z:
            for j in range(gpos.size(1)):
                z_lines = np.stack([
                    np.stack((start[:, j, 0], start[:, j, 1]+0.0015*i, start[:, j, 2], z_end[:, j, 0], z_end[:, j, 1]+0.0015*i, z_end[:, j, 2]), -1)
                            for i in range(n_lines)], -2)                                      # [n_envs, n_lines, 6]
                for e, l in zip(self.envs, z_lines):
                    self.gym.add_lines(self.viewer, e, n_lines, l, [[0., 0., 1.] for _ in range(n_lines)])
        pass
    
    def get_link_len(self, p_idx, c_idx):
        p_pos, c_pos = self.link_pos[0, p_idx, :], self.link_pos[0, c_idx, :] # [n_links, 3]
        link_len = torch.linalg.norm((p_pos - c_pos), ord=2, dim=-1, keepdim=True)  # [n_links, 1]
        return link_len 
        # what I added
    def create_motion_info(self):
        pass
        # what I added
@torch.jit.script
def observe_iccgan(state_hist: torch.Tensor, seq_len: torch.Tensor):
    # state_hist: L x N x D

    UP_AXIS = 2
    n_hist = state_hist.size(0)
    n_inst = state_hist.size(1)

    root_tensor = state_hist[..., :13]
    link_tensor = state_hist[...,13:].view(state_hist.size(0), state_hist.size(1), -1, 13)

    root_pos, root_orient = root_tensor[..., :3], root_tensor[..., 3:7]
    link_pos, link_orient = link_tensor[..., :3], link_tensor[..., 3:7]
    link_lin_vel, link_ang_vel = link_tensor[..., 7:10], link_tensor[..., 10:13]

    origin = root_pos[-1].clone()
    origin[..., UP_AXIS] = 0                                            # N x 3
    heading = heading_zup(root_orient[-1])
    up_dir = torch.zeros_like(origin)
    up_dir[..., UP_AXIS] = 1
    heading_orient_inv = axang2quat(up_dir, -heading)                   # N x 4

    heading_orient_inv = (heading_orient_inv                            # L x N x n_links x 4
        .view(1, -1, 1, 4).repeat(n_hist, 1, link_pos.size(-2), 1))
    origin = origin.unsqueeze_(-2)                                      # N x 1 x 3

    ob_link_pos = link_pos - origin                                     # L x N x n_links x 3 
    ob_link_pos = rotatepoint(heading_orient_inv, ob_link_pos)
    ob_link_orient = quatmultiply(heading_orient_inv, link_orient)      # L x N x n_links x 4
    ob_link_lin_vel = rotatepoint(heading_orient_inv, link_lin_vel)     # N x n_links x 3
    ob_link_ang_vel = rotatepoint(heading_orient_inv, link_ang_vel)     # N x n_links x 3

    ob = torch.cat((ob_link_pos, ob_link_orient,
        ob_link_lin_vel, ob_link_ang_vel), -1)                          # L x N x n_links x 13
    ob = ob.view(n_hist, n_inst, -1)                                    # L x N x (n_links x 13)

    ob1 = ob.permute(1, 0, 2)                                           # N x L x (n_links x 13)
    ob2 = torch.zeros_like(ob1)
    arange = torch.arange(n_hist, dtype=seq_len.dtype, device=seq_len.device).unsqueeze_(0)
    seq_len_ = seq_len.unsqueeze(1)
    mask1 = arange > (n_hist-1) - seq_len_
    mask2 = arange < seq_len_
    ob2[mask2] = ob1[mask1]
    return ob2.flatten(start_dim=1)


@torch.jit.script
def observe_disc(state_hist: torch.Tensor, seq_len: torch.Tensor, key_links: Optional[List[int]]=None,
    parent_link: Optional[int]=None, local_pos: bool=False, local_height: bool = True,
):
    # state_hist: L x N x D

    UP_AXIS = 2
    n_hist = state_hist.size(0)
    n_inst = state_hist.size(1)

    root_tensor = state_hist[..., :13]
    link_tensor = state_hist[...,13:].view(n_hist, n_inst, -1, 13)
    if key_links is None:
        link_pos, link_orient = link_tensor[...,:3], link_tensor[...,3:7]
    else:
        link_pos, link_orient = link_tensor[:,:,key_links,:3], link_tensor[:,:,key_links,3:7]

    if parent_link is None:
        origin = root_tensor[-1,:, :3].clone()               # N x 3
        origin[..., UP_AXIS] = 0
        orient = root_tensor[-1,:,3:7]                    # N x 4
    else:
        origin = link_tensor[:,:, parent_link, :3]      # L x N x 3
        orient = link_tensor[:,:, parent_link,3:7]      # L x N x 4
        if not local_height:
            origin = origin.clone()
            origin[..., UP_AXIS] = 0

    if local_pos:
        orient_inv = quatconj(orient)               # N x 4 or L x N x 4
    else:
        heading = heading_zup(orient)
        up_dir = torch.zeros_like(origin)
        up_dir[..., UP_AXIS] = 1
        orient_inv = axang2quat(up_dir, -heading)
    origin.unsqueeze_(-2)                           # N x 1 x 3 or L x N x 1 x 3

    if parent_link is None:
        orient_inv = orient_inv.view(1, -1, 1, 4)  # 1 x N x 1 x 4
    else:
        orient_inv = orient_inv.unsqueeze_(-2)     # L x N x 1 x 4

    ob_link_pos = link_pos - origin                                     # L x N x n_links x 3 
    ob_link_pos = rotatepoint(orient_inv, ob_link_pos)
    ob_link_orient = quatmultiply(orient_inv, link_orient)              # L x N x n_links x 4

    ob = torch.cat((ob_link_pos, ob_link_orient), -1)                   # L x N x n_links x 7
    ob = ob.view(n_hist, n_inst, -1)                                    # L x N x (n_links x 7)

    ob1 = ob.permute(1, 0, 2)                                           # N x L x (n_links x 7)
    ob2 = torch.zeros_like(ob1)
    arange = torch.arange(n_hist, dtype=seq_len.dtype, device=seq_len.device).unsqueeze_(0)
    seq_len_ = seq_len.unsqueeze(1)
    mask1 = arange > (n_hist-1) - seq_len_
    mask2 = arange < seq_len_
    ob2[mask2] = ob1[mask1]
    
    return ob2



class ICCGANHumanoidTarget(ICCGANHumanoid):

    GOAL_REWARD_WEIGHT = 0.5
    GOAL_DIM = 4                    # (x, y, sp, dist)
    GOAL_TENSOR_DIM = 3             # global position of root target (X, Y, Z) - where root should reach
    ENABLE_GOAL_TIMER = True

    GOAL_RADIUS = 0.5
    SP_LOWER_BOUND = 1.2
    SP_UPPER_BOUND = 1.5
    GOAL_TIMER_RANGE = 90, 150
    GOAL_SP_MEAN = 1
    GOAL_SP_STD = 0.25
    GOAL_SP_MIN = 0
    GOAL_SP_MAX = 1.25

    SHARP_TURN_RATE = 1

    def __init__(self, *args, **kwargs):
        self.goal_radius = parse_kwarg(kwargs, "goal_radius", self.GOAL_RADIUS)
        self.sharp_turn_rate = parse_kwarg(kwargs, "sharp_turn_rate", self.SHARP_TURN_RATE)
        self.sp_lower_bound = parse_kwarg(kwargs, "sp_lower_bound", self.SP_LOWER_BOUND)
        self.sp_upper_bound = parse_kwarg(kwargs, "sp_upper_bound", self.SP_UPPER_BOUND)
        self.goal_timer_range = parse_kwarg(kwargs, "goal_timer_range", self.GOAL_TIMER_RANGE)
        self.goal_sp_mean = parse_kwarg(kwargs, "goal_sp_mean", self.GOAL_SP_MEAN)
        self.goal_sp_std = parse_kwarg(kwargs, "goal_sp_std", self.GOAL_SP_STD)
        self.goal_sp_min = parse_kwarg(kwargs, "goal_sp_min", self.GOAL_SP_MIN)
        self.goal_sp_max = parse_kwarg(kwargs, "goal_sp_max", self.GOAL_SP_MAX)
        super().__init__(*args, **kwargs)

    def update_viewer(self):
        super().update_viewer()
        self.gym.clear_lines(self.viewer)
        n_lines = 10
        tar_x = self.goal_tensor[:, 0].cpu().numpy()

        p = self.root_pos.cpu().numpy()
        zero = np.zeros_like(tar_x)+0.05
        tar_y = self.goal_tensor[:, 1].cpu().numpy()
        lines = np.stack([
            np.stack((p[:,0], p[:,1], zero+0.01*i, tar_x, tar_y, zero), -1)
        for i in range(n_lines)], -2)
        for e, l in zip(self.envs, lines):
            self.gym.add_lines(self.viewer, e, n_lines, l, [[1., 0., 0.] for _ in range(n_lines)])  # red
        n_lines = 10
        target_pos = self.goal_tensor.cpu().numpy()
        lines = np.stack([
            np.stack((
                target_pos[:, 0], target_pos[:, 1], zero,
                target_pos[:, 0]+self.goal_radius*np.cos(2*np.pi/n_lines*i), 
                target_pos[:, 1]+self.goal_radius*np.sin(2*np.pi/n_lines*i),
                zero
            ), -1)
        for i in range(n_lines)], -2)
        for e, l in zip(self.envs, lines):
            self.gym.add_lines(self.viewer, e, n_lines, l, [[0., 0., 1.] for _ in range(n_lines)])  # blue
    
    def _observe(self, env_ids):
        if env_ids is None:
            return observe_iccgan_target(
                self.state_hist[-self.ob_horizon:], self.ob_seq_lens,
                self.goal_tensor, self.goal_timer, sp_upper_bound=self.sp_upper_bound, fps=self.fps
            )
        else:
            return observe_iccgan_target(
                self.state_hist[-self.ob_horizon:][:, env_ids], self.ob_seq_lens[env_ids],
                self.goal_tensor[env_ids], self.goal_timer[env_ids], sp_upper_bound=self.sp_upper_bound, fps=self.fps
            )

    def reset_goal(self, env_ids, goal_tensor=None, goal_timer=None):
        #! shallow copy: 이렇게 되면 goal_tensor가 바뀌면 self.goal_tensor도 바뀐다!
        if goal_tensor is None: goal_tensor = self.goal_tensor
        if goal_timer is None: goal_timer = self.goal_timer
        
        n_envs = len(env_ids)
        all_envs = n_envs == len(self.envs)
        root_orient = self.root_orient if all_envs else self.root_orient[env_ids]

        small_turn = torch.rand(n_envs, device=self.device) > self.sharp_turn_rate                      # 0~1 사이 난수 발생
        large_angle = torch.rand(n_envs, dtype=torch.float32, device=self.device).mul_(2*np.pi)         # 0~2pi 사이
        small_angle = torch.rand(n_envs, dtype=torch.float32, device=self.device).sub_(0.5).mul_(2*(np.pi/3))   

        heading = heading_zup(root_orient)
        small_angle += heading
        theta = torch.where(small_turn, small_angle, large_angle)   # (condition, input, other)

        timer = torch.randint(self.goal_timer_range[0], self.goal_timer_range[1], (n_envs,), dtype=self.goal_timer.dtype, device=self.device)
        if self.goal_sp_min == self.goal_sp_max:     # juggling+locomotion_walk
            vel = self.goal_sp_min
        elif self.goal_sp_std == 0:                  # juggling+locomotion_walk
            vel = self.goal_sp_mean
        else:
            vel = torch.nn.init.trunc_normal_(torch.empty(n_envs, dtype=torch.float32, device=self.device), mean=self.goal_sp_mean, std=self.goal_sp_std, a=self.goal_sp_min, b=self.goal_sp_max)
        
        dist = vel*timer*self.step_time     # 1/fps에서 얼만큼 갈 수 있는가
        dx = dist*torch.cos(theta)
        dy = dist*torch.sin(theta)

        if all_envs:
            self.init_dist = dist
            goal_timer.copy_(timer)
            goal_tensor[:,0] = self.root_pos[:,0] + dx
            goal_tensor[:,1] = self.root_pos[:,1] + dy
        else:
            self.init_dist[env_ids] = dist
            goal_timer[env_ids] = timer
            goal_tensor[env_ids,0] = self.root_pos[env_ids,0] + dx
            goal_tensor[env_ids,1] = self.root_pos[env_ids,1] + dy
        
    def reward(self, goal_tensor=None, goal_timer=None):
        if goal_tensor is None: goal_tensor = self.goal_tensor
        if goal_timer is None: goal_timer = self.goal_timer

        p = self.root_pos                                       # 현재 root_pos
        p_ = self.state_hist[-1][:, :3]                         # 이전 root_pos (goal_tensor 구했을 때의 root_pos부터 시작!  / action apply 되기 이전)

        dp_ = goal_tensor - p_                                  # root_pos에서 target 지점까지의 global (dx, dy)
        dp_[:, self.UP_AXIS] = 0
        dist_ = torch.linalg.norm(dp_, ord=2, dim=-1)
        v_ = dp_.div_(goal_timer.unsqueeze(-1)*self.step_time)  # v_: desired veloicty (total distance / sec)

        v_mag = torch.linalg.norm(v_, ord=2, dim=-1)
        sp_ = (dist_/self.step_time).clip_(max=v_mag.clip(min=self.sp_lower_bound, max=self.sp_upper_bound))
        v_ *= (sp_/v_mag).unsqueeze_(-1)                       # desired velocity

        dp = p - p_                                            # (현재 root - 이전 root)
        dp[:, self.UP_AXIS] = 0
        v = dp / self.step_time                                # current velocity: dp / duration 
        r = (v - v_).square_().sum(1).mul_(-3/(sp_*sp_)).exp_()

        dp = goal_tensor - p
        dp[:, self.UP_AXIS] = 0
        dist = torch.linalg.norm(dp, ord=2, dim=-1)
        self.near = dist < self.goal_radius

        r[self.near] = 1
        
        if self.viewer is not None:
            self.goal_timer[self.near] = self.goal_timer[self.near].clip(max=20)
        
        return r.unsqueeze_(-1)

    def termination_check(self, goal_tensor=None):
        if goal_tensor is None: goal_tensor = self.goal_tensor
        fall = super().termination_check()
        dp = goal_tensor - self.root_pos
        dp[:, self.UP_AXIS] = 0
        dist = dp.square_().sum(-1).sqrt_()
        too_far = dist-self.init_dist > 3
        return torch.logical_or(fall, too_far)


@torch.jit.script
def observe_iccgan_target(state_hist: torch.Tensor, seq_len: torch.Tensor,
    target_tensor: torch.Tensor, timer: torch.Tensor,
    sp_upper_bound: float, fps: int
):
    ob = observe_iccgan(state_hist, seq_len)

    root_pos = state_hist[-1, :, :3]
    root_orient = state_hist[-1, :, 3:7]

    dp = target_tensor - root_pos
    x = dp[:, 0]
    y = dp[:, 1]
    heading_inv = -heading_zup(root_orient)
    c = torch.cos(heading_inv)      # root_orientation의 x-dir의 각도 (inverse) 
    s = torch.sin(heading_inv)
    x, y = c*x-s*y, s*x+c*y         # [[c -s], [s c]] * [x y]^T (local_dp -> root_orient에서 바라본 dp)

    dist = (x*x + y*y).sqrt_()
    sp = dist.mul(fps/timer)        # speed! ... dist/timer->how many dist we should go per step ... dist*fps/timer -> how much distance we should go in 1 sec

    too_close = dist < 1e-5
    x = torch.where(too_close, x, x/dist)   # x/dist: normalized x
    y = torch.where(too_close, y, y/dist)
    sp.clip_(max=sp_upper_bound)
    dist.div_(3).clip_(max=1.5)

    return torch.cat((ob, x.unsqueeze_(-1), y.unsqueeze_(-1), sp.unsqueeze_(-1), dist.unsqueeze_(-1)), -1)



class ICCGANHumanoidTargetAiming(ICCGANHumanoidTarget):
    
    GOAL_REWARD_WEIGHT = 0.25, 0.25
    GOAL_DIM = 4+3
    GOAL_TENSOR_DIM = 3+4

    def create_tensors(self):
        super().create_tensors()
        self.hand_link = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actors[0], "right_hand")
        self.lower_arm_link = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actors[0], "right_lower_arm")
        self.aiming_start_link = self.lower_arm_link
        self.aiming_end_link = self.hand_link


        self.x_dir = torch.zeros_like(self.root_pos)
        self.x_dir[..., 0] = 1
        self.reverse_rotation = torch.zeros_like(self.root_orient)
        self.reverse_rotation[..., self.UP_AXIS] = 1

    def _observe(self, env_ids):
        if env_ids is None:
            return observe_iccgan_target_aiming(
                self.state_hist[-self.ob_horizon:], self.ob_seq_lens,
                self.goal_tensor, self.goal_timer,
                sp_upper_bound=self.sp_upper_bound, goal_radius=self.goal_radius, fps=self.fps
            )
        else:
            return observe_iccgan_target_aiming(
                self.state_hist[-self.ob_horizon:][:, env_ids], self.ob_seq_lens[env_ids],
                self.goal_tensor[env_ids], self.goal_timer[env_ids],
                sp_upper_bound=self.sp_upper_bound, goal_radius=self.goal_radius, fps=self.fps
            )

    def update_viewer(self):
        super().update_viewer()

        target_tensor = self.goal_tensor[:, :3]
        aiming_tensor = self.goal_tensor[:, 3:]

        target_dir = target_tensor - self.root_pos
        target_dir[..., self.UP_AXIS] = 0
        dist = torch.linalg.norm(target_dir, ord=2, dim=-1, keepdim=True)
        not_near = (dist > self.goal_radius).squeeze_(-1)
        dist = dist[not_near]

        if dist.nelement() < 1: return

        target_dir = target_dir[not_near]
        target_dir.div_(dist)
        link_pos = self.link_pos[not_near]

        x_dir = self.x_dir[:target_dir.size(0)]
        q = quatdiff_normalized(x_dir, target_dir)
        # ensure 180 degree rotation is around the up axis
        q = torch.where(target_dir[:, :1] < -0.99999,
            self.reverse_rotation, q)

        aiming_dir = rotatepoint(quatmultiply(q, aiming_tensor), x_dir)

        start = link_pos[:, self.aiming_start_link]
        end = start + aiming_dir

        start = start.cpu().numpy()
        end = end.cpu().numpy()
        not_near = torch.nonzero(not_near).view(-1).cpu().numpy()
        n_lines = 10
        lines = np.stack([
            np.stack((start[:,0], start[:,1], start[:,2]+0.005*i, end[:, 0], end[:, 1], end[:,2]+0.005*i), -1)
        for i in range(-n_lines//2, n_lines//2)], -2)
        for i, l in zip(not_near, lines):
            e = self.envs[i]
            self.gym.add_lines(self.viewer, e, n_lines, l, [[0., 1., 0.] for _ in range(n_lines)])
            
    def reset_goal(self, env_ids):
        super().reset_goal(env_ids, self.goal_tensor[:, :3])
        self.reset_aiming_goal(env_ids)
    
    def reset_aiming_goal(self, env_ids):
        n_envs = len(env_ids)
        elev = torch.rand(n_envs, dtype=torch.float32, device=self.device).mul_(-np.pi/6)
        azim = torch.rand(n_envs, dtype=torch.float32, device=self.device).mul_(np.pi/4)
        if self.viewer is not None: azim.add_(0.3)

        elev /= 2
        azim /= 2
        cp = torch.cos(elev) # y
        sp = torch.sin(elev)
        cy = torch.cos(azim) # z
        sy = torch.sin(azim)

        w = cp*cy  # cr*cp*cy + sr*sp*sy
        x = -sp*sy # sr*cp*cy - cr*sp*sy
        y = sp*cy  # cr*sp*cy + sr*cp*sy
        z = cp*sy  # cr*cp*sy - sr*sp*cy
        
        if n_envs == len(self.envs):
            self.goal_tensor[:, 3] = x
            self.goal_tensor[:, 4] = y
            self.goal_tensor[:, 5] = z 
            self.goal_tensor[:, 6] = w
        else:
            self.goal_tensor[env_ids, 3] = x
            self.goal_tensor[env_ids, 4] = y
            self.goal_tensor[env_ids, 5] = z
            self.goal_tensor[env_ids, 6] = w

    def reward(self):
        target_tensor = self.goal_tensor[:, :3]
        aiming_tensor = self.goal_tensor[:, 3:]
        
        target_rew = super().reward(target_tensor)

        dp = target_tensor - self.root_pos
        dp[..., self.UP_AXIS] = 0
        dist = torch.linalg.norm(dp, ord=2, dim=-1, keepdim=True)
        
        target_dir = dp / dist
        q0 = quatdiff_normalized(self.x_dir, target_dir)
        q = torch.where(target_dir[:, :1] < -0.99999,
            self.reverse_rotation, q0)

        aiming_dir = rotatepoint(quatmultiply(q, aiming_tensor), self.x_dir)

        hand_pos = self.link_pos[:, self.aiming_end_link]
        fore_arm_pos = self.link_pos[:, self.aiming_start_link]

        fore_arm_dir = hand_pos - fore_arm_pos
        arm_len = torch.linalg.norm(fore_arm_dir, ord=2, dim=-1, keepdim=True)
        fore_arm_dir.div_(arm_len)

        target_hand_pos = fore_arm_pos + arm_len * aiming_dir
        e = torch.linalg.norm(target_hand_pos.sub_(hand_pos), ord=2, dim=-1).div_(arm_len.squeeze_(-1))
        aiming_rew = e.mul_(-2).exp_()

        rest_rew = fore_arm_dir[..., self.UP_AXIS].div(0.8).clip_(min=0, max=1) # 2nd reward to encourage character to lift its arm when aiming action is not activated
        
        aiming_rew = torch.where(self.near, rest_rew, aiming_rew).unsqueeze_(-1)

        r = torch.cat((target_rew, aiming_rew), -1)
        return r

    def termination_check(self):
        return super().termination_check(self.goal_tensor[:, :3])


@torch.jit.script
def observe_iccgan_target_aiming(state_hist: torch.Tensor, seq_len: torch.Tensor, 
    goal_tensor: torch.Tensor, timer: torch.Tensor,
    sp_upper_bound: float, goal_radius: float, fps: int
):
    UP_AXIS = 2

    target_tensor = goal_tensor[..., :3]
    aiming_tensor = goal_tensor[..., 3:]

    target_ob = observe_iccgan_target(state_hist, seq_len, target_tensor, timer, sp_upper_bound=sp_upper_bound, fps=fps)
    
    root_pos = state_hist[-1, :, :3]
    root_orient = state_hist[-1, :, 3:7]
    heading = heading_zup(root_orient)
    up_dir = torch.zeros_like(root_pos)
    up_dir[..., UP_AXIS] = 1
    orient_inv = axang2quat(up_dir, -heading)

    dp = target_tensor - root_pos
    dp[..., UP_AXIS] = 0
    dist = torch.linalg.norm(dp, ord=2, dim=-1, keepdim=True)
    
    x_dir = torch.zeros_like(dp)
    x_dir[..., 0] = 1
    target_dir = dp / dist
    q = quatdiff_normalized(x_dir, target_dir)

    # ensure 180 degree rotation is around the up axis
    reverse = torch.zeros_like(q)
    reverse[..., UP_AXIS] = 1
    q = torch.where(target_dir[:, :1] < -0.99999,
        reverse, q)

    aiming_dir = quatmultiply(q, aiming_tensor)
    aiming_dir = quatmultiply(q, aiming_tensor)
    aiming_dir = quatmultiply(orient_inv, aiming_dir)
    aiming_dir = rotatepoint(aiming_dir, x_dir)

    near = dist.squeeze_(-1) < goal_radius
    # aiming_dir[near] = 0 # not supported by script
    aiming_dir[near, 0] = 0
    aiming_dir[near, 1] = 0
    aiming_dir[near, 2] = 0
    
    return torch.cat((target_ob, aiming_dir), -1)

class ICCGANHumanoidEE(ICCGANHumanoid):

    GOAL_REWARD_WEIGHT = 0.5
    GOAL_DIM = 4 + 4                # (x, y, z, dist)
    GOAL_TENSOR_DIM = 3 + 3         # global position of rhand target (X, Y, Z) - where rhand should reach
    ENABLE_GOAL_TIMER = True

    GOAL_RADIUS = 0.5
    SP_LOWER_BOUND = 1.2
    SP_UPPER_BOUND = 1.5
    GOAL_TIMER_RANGE = 90, 150
    GOAL_SP_MEAN = 1
    GOAL_SP_STD = 0.25
    GOAL_SP_MIN = 0
    GOAL_SP_MAX = 1.25

    SHARP_TURN_RATE = 1

    def __init__(self, *args, **kwargs):
        self.goal_radius = parse_kwarg(kwargs, "goal_radius", self.GOAL_RADIUS)
        self.sharp_turn_rate = parse_kwarg(kwargs, "sharp_turn_rate", self.SHARP_TURN_RATE)
        self.sp_lower_bound = parse_kwarg(kwargs, "sp_lower_bound", self.SP_LOWER_BOUND)
        self.sp_upper_bound = parse_kwarg(kwargs, "sp_upper_bound", self.SP_UPPER_BOUND)
        self.goal_timer_range = parse_kwarg(kwargs, "goal_timer_range", self.GOAL_TIMER_RANGE)
        self.goal_sp_mean = parse_kwarg(kwargs, "goal_sp_mean", self.GOAL_SP_MEAN)
        self.goal_sp_std = parse_kwarg(kwargs, "goal_sp_std", self.GOAL_SP_STD)
        self.goal_sp_min = parse_kwarg(kwargs, "goal_sp_min", self.GOAL_SP_MIN)
        self.goal_sp_max = parse_kwarg(kwargs, "goal_sp_max", self.GOAL_SP_MAX)
        super().__init__(*args, **kwargs)

        # get_link_len
        rarm_len, larm_len = self.get_link_len([2,3,4], [3,4,5]), self.get_link_len([2,6,7], [6,7,8])
        self.rarm_len, self.larm_len = rarm_len.sum(dim=0), larm_len.sum(dim=0)

    def create_tensors(self):
        super().create_tensors()
        self.r_hand_link = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actors[0], "right_hand")
        self.head = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actors[0], "head")
        self.aiming_start_link = self.head
        self.r_aiming_end_link = self.r_hand_link

        # left hand added
        self.l_hand_link = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actors[0], "left_hand")
        self.l_aiming_end_link = self.l_hand_link

        self.x_dir = torch.zeros_like(self.root_pos)
        self.x_dir[..., 0] = 1
        self.reverse_rotation = torch.zeros_like(self.root_orient)
        self.reverse_rotation[..., self.UP_AXIS] = 1

    def update_viewer(self):
        super().update_viewer()
        self.gym.clear_lines(self.viewer)

        # #! what I added for ee position
        # # 1. rhand visualization
        # link_pos = self.link_pos
        # r_end = self.goal_tensor[:, 0:3].cpu().numpy()

        # for i in range(len(self.envs)):
        #     rsphere_geom = gymutil.WireframeSphereGeometry(0.04, 16, 16, None, color=(1, 0, 0))   # red
        #     rhand_pos = r_end[i]
        #     rhand_pose = gymapi.Transform(gymapi.Vec3(rhand_pos[0], rhand_pos[1], rhand_pos[2]), r=None)
        #     gymutil.draw_lines(rsphere_geom, self.gym, self.viewer, self.envs[i], rhand_pose)   
        # #!

        # # 2. lhand visualization
        # l_end = self.ltemp.cpu().numpy()

        # for i in range(len(self.envs)):
        #     lsphere_geom = gymutil.WireframeSphereGeometry(0.04, 16, 16, None, color=(0, 1, 0))   # green
        #     lhand_pos = l_end[i]
        #     lhand_pose = gymapi.Transform(gymapi.Vec3(lhand_pos[0], lhand_pos[1], lhand_pos[2]), r=None)
        #     gymutil.draw_lines(lsphere_geom, self.gym, self.viewer, self.envs[i], lhand_pose)   
        # #!

    def _observe(self, env_ids):
        if env_ids is None:
            return observe_iccgan_ee(
                self.state_hist[-self.ob_horizon:], self.ob_seq_lens,
                self.goal_tensor, self.goal_timer, sp_upper_bound=self.sp_upper_bound, fps=self.fps
            )
        else:
            return observe_iccgan_ee(
                self.state_hist[-self.ob_horizon:][:, env_ids], self.ob_seq_lens[env_ids],
                self.goal_tensor[env_ids], self.goal_timer[env_ids], sp_upper_bound=self.sp_upper_bound, fps=self.fps
            )

    def reset_goal(self, env_ids, goal_tensor=None, goal_timer=None):
        #! shallow copy: 이렇게 되면 goal_tensor가 바뀌면 self.goal_tensor도 바뀐다!
        if goal_tensor is None: goal_tensor = self.goal_tensor
        if goal_timer is None: goal_timer = self.goal_timer
        
        n_envs = len(env_ids)
        all_envs = n_envs == len(self.envs)
        root_orient = self.root_orient if all_envs else self.root_orient[env_ids]

        small_turn = torch.rand(n_envs, device=self.device) > self.sharp_turn_rate
        large_angle = torch.rand(n_envs, dtype=torch.float32, device=self.device).mul_(2*np.pi)
        small_angle = torch.rand(n_envs, dtype=torch.float32, device=self.device).sub_(0.5).mul_(2*(np.pi/3))

        heading = heading_zup(root_orient)
        small_angle += heading
        theta = torch.where(small_turn, small_angle, large_angle)

        timer = torch.randint(self.goal_timer_range[0], self.goal_timer_range[1], (n_envs,), dtype=self.goal_timer.dtype, device=self.device)
        if self.goal_sp_min == self.goal_sp_max:     # juggling+locomotion_walk
            vel = self.goal_sp_min
        elif self.goal_sp_std == 0:                  # juggling+locomotion_walk
            vel = self.goal_sp_mean
        else:
            vel = torch.nn.init.trunc_normal_(torch.empty(n_envs, dtype=torch.float32, device=self.device), mean=self.goal_sp_mean, std=self.goal_sp_std, a=self.goal_sp_min, b=self.goal_sp_max)
        
        dist = vel*timer*self.step_time     # 1/fps에서 얼만큼 갈 수 있는가
        dx = dist*torch.cos(theta)
        dy = dist*torch.sin(theta)

        if all_envs:
            self.init_dist = dist
            goal_timer.copy_(timer)
            # goal_tensor[:,0] = self.root_pos[:,0] + dx
            # goal_tensor[:,1] = self.root_pos[:,1] + dy
        else:
            self.init_dist[env_ids] = dist
            goal_timer[env_ids] = timer
            # goal_tensor[env_ids,0] = self.root_pos[env_ids,0] + dx
            # goal_tensor[env_ids,1] = self.root_pos[env_ids,1] + dy
        

        #! what I added for ee position
        n_envs = len(env_ids)
        elev = torch.rand(n_envs, dtype=torch.float32, device=self.device).mul_(-np.pi/2)
        azim = torch.ones(n_envs, dtype=torch.float32, device=self.device).mul_(-np.pi/2)
        elev /= 2
        azim /= 2
        cp = torch.cos(elev) # y
        sp = torch.sin(elev)
        cy = torch.cos(azim) # z
        sy = torch.sin(azim)

        w = cp*cy  # cr*cp*cy + sr*sp*sy
        x = -sp*sy # sr*cp*cy - cr*sp*sy
        y = sp*cy  # cr*sp*cy + sr*cp*sy
        z = cp*sy  # cr*cp*sy - sr*sp*cy

        self.temp = torch.zeros((len(self.envs), 4), device=self.device)

        if n_envs == len(self.envs):
            self.temp[:, 0] = x
            self.temp[:, 1] = y
            self.temp[:, 2] = z 
            self.temp[:, 3] = w
        else:
            self.temp[env_ids, 0] = x
            self.temp[env_ids, 1] = y
            self.temp[env_ids, 2] = z
            self.temp[env_ids, 3] = w     
        #!

        #! what I added for ee position
        n_envs = len(env_ids)
        l_elev = torch.rand(n_envs, dtype=torch.float32, device=self.device).mul_(-np.pi/2)
        l_azim = torch.ones(n_envs, dtype=torch.float32, device=self.device).mul_(np.pi/2)
        l_elev /= 2
        l_azim /= 2
        l_cp = torch.cos(l_elev) # y
        l_sp = torch.sin(l_elev)
        l_cy = torch.cos(l_azim) # z
        l_sy = torch.sin(l_azim)

        l_w = l_cp*l_cy  # cr*cp*cy + sr*sp*sy
        l_x = -l_sp*l_sy # sr*cp*cy - cr*sp*sy
        l_y = l_sp*l_cy  # cr*sp*cy + sr*cp*sy
        l_z = l_cp*l_sy  # cr*cp*sy - sr*sp*cy

        self.temp2 = torch.zeros((len(self.envs), 4), device=self.device)

        if n_envs == len(self.envs):
            self.temp2[:, 0] = l_x
            self.temp2[:, 1] = l_y
            self.temp2[:, 2] = l_z 
            self.temp2[:, 3] = l_w
        else:
            self.temp2[env_ids, 0] = l_x
            self.temp2[env_ids, 1] = l_y
            self.temp2[env_ids, 2] = l_z
            self.temp2[env_ids, 3] = l_w     
        #!

        #! what I added for ee position
        # 1. calculate root_heading_dir as target_dir
        root_orient = self.root_orient[env_ids]
        
        root_gheading_dir = torch.zeros_like(root_orient[...,:3])         # (n_envs, 3)
        root_gheading_dir[..., 0] = 1
        root_gheading_dir = rotatepoint(root_orient, root_gheading_dir)   #! heading은 root! global root heading direction

        root_gheading_dir[..., self.UP_AXIS] = 0                    
        dist = torch.linalg.norm(root_gheading_dir, ord=2, dim=-1, keepdim=True)    # (n_envs, 3)
        
        root_gheading_dir.div_(dist)

        link_pos = self.link_pos[env_ids]                                           # (n_envs, 3)                                               
        reverse_rotation = self.reverse_rotation[env_ids]                           # (n_envs, 3)

        x_dir = torch.zeros_like(root_orient[..., :3])                              # (n_envs, 3)
        x_dir[..., 0] = 1
        x_dir = x_dir[:root_gheading_dir.size(0)]
        q = quatdiff_normalized(x_dir, root_gheading_dir)                 # global x-axis에서 root_gheading_dir까지의 quaternion representation of the rotation 

        q = torch.where(root_gheading_dir[:, :1] < -0.99999,              # root_gheading_dir이 (-1,0,0)이면 그냥 q=(0,0,1,0)
            reverse_rotation, q)

        # 2. rhand_aiming_tensor to rhand_aiming_dir
        rhand_aiming_tensor = self.temp[env_ids]                            # (n_envs, 3)
        rhand_aiming_dir = rotatepoint(quatmultiply(q, rhand_aiming_tensor), x_dir)   # GLOBAL rhand_aiming_dir (x-dir) # (n_envs, 3)
        dist = torch.linalg.norm(rhand_aiming_dir, ord=2, dim=-1, keepdim=True)
        rhand_aiming_dir.div_(dist)                                                   # normalize dir                     

        start = link_pos[:, self.aiming_start_link]                        #! start는 head
        rarm_offset, larm_offset = self.rarm_len.item(), self.larm_len.item()
        r_end = start + rhand_aiming_dir * rarm_offset                     #! shape: (n_envs, 3)

        # self.temp2 = torch.zeros((n_envs, 3), device=self.device)
        if n_envs == len(self.envs):
            goal_tensor[:, 0] = r_end[:, 0]
            goal_tensor[:, 1] = r_end[:, 1]
            goal_tensor[:, 2] = r_end[:, 2]
        else:
            goal_tensor[env_ids, 0] = r_end[:, 0]
            goal_tensor[env_ids, 1] = r_end[:, 1]
            goal_tensor[env_ids, 2] = r_end[:, 2]
        #!hum

        #! ADDED lhand position
        lhand_aiming_tensor = self.temp2[env_ids]
        lhand_aiming_dir = rotatepoint(quatmultiply(q, lhand_aiming_tensor), x_dir)   # GLOBAL lhand_aiming_dir (x-dir)
        l_dist = torch.linalg.norm(lhand_aiming_dir, ord=2, dim=-1, keepdim=True)
        lhand_aiming_dir.div_(l_dist)                                                   # normalize dir                     

        l_end = start + lhand_aiming_dir * larm_offset                     #! shape: (n_envs, 3)

        self.ltemp = torch.zeros((len(self.envs), 3), device=self.device)
        if n_envs == len(self.envs):
            self.ltemp[:, 0] = l_end[:, 0]
            self.ltemp[:, 1] = l_end[:, 1]
            self.ltemp[:, 2] = l_end[:, 2]
        else:
            self.ltemp[env_ids, 0] = l_end[:, 0]
            self.ltemp[env_ids, 1] = l_end[:, 1]
            self.ltemp[env_ids, 2] = l_end[:, 2]
        #!
    def reward(self, goal_tensor=None, goal_timer=None):
        if goal_tensor is None: goal_tensor = self.goal_tensor
        if goal_timer is None: goal_timer = self.goal_timer

        p = self.root_pos                                       # 현재 root_pos
        p_ = self.state_hist[-1][:, :3]                         # 이전 root_pos (goal_tensor 구했을 때의 root_pos부터 시작!  / action apply 되기 이전)

        dp_ = goal_tensor[..., :3] - p_                                  # root_pos에서 target 지점까지의 global (dx, dy)
        dp_[:, self.UP_AXIS] = 0
        dist_ = torch.linalg.norm(dp_, ord=2, dim=-1)
        v_ = dp_.div_(goal_timer.unsqueeze(-1)*self.step_time)  # v_: desired veloicty (total distance / sec)

        v_mag = torch.linalg.norm(v_, ord=2, dim=-1)
        sp_ = (dist_/self.step_time).clip_(max=v_mag.clip(min=self.sp_lower_bound, max=self.sp_upper_bound))
        v_ *= (sp_/v_mag).unsqueeze_(-1)                       # desired velocity

        dp = p - p_                                            # (현재 root - 이전 root)
        dp[:, self.UP_AXIS] = 0
        v = dp / self.step_time                                # current velocity: dp / duration 
        r = (v - v_).square_().sum(1).mul_(-3/(sp_*sp_)).exp_()

        dp = goal_tensor[..., :3] - p
        dp[:, self.UP_AXIS] = 0
        dist = torch.linalg.norm(dp, ord=2, dim=-1)
        self.near = dist < self.goal_radius

        r[self.near] = 1
        
        if self.viewer is not None:
            self.goal_timer[self.near] = self.goal_timer[self.near].clip(max=20)
        
        #! I added
        target_rhand_pos = goal_tensor[..., :3]
        rhand_pos = self.link_pos[:, self.r_aiming_end_link]
        dp_hand = target_rhand_pos - rhand_pos

        rarm_len, larm_len = self.get_link_len([2,3,4], [3,4,5]), self.get_link_len([2,6,7], [6,7,8])
        rarm_len, larm_len = rarm_len.sum(dim=0), larm_len.sum(dim=0)
        rarm_len = rarm_len.repeat(len(self.envs))
        
        #! 이건 near 생각해보기!
        dist_hand = torch.linalg.norm(dp_hand, ord=2, dim=-1)
        e = torch.linalg.norm(target_rhand_pos.sub(rhand_pos), ord=2, dim=-1).div_(rarm_len)
        rhand_rew = e.mul_(-2).exp_()

        #! I added
        # return r.unsqueeze_(-1)
        return rhand_rew.unsqueeze_(-1)

    def termination_check(self, goal_tensor=None):
        # if goal_tensor is None: goal_tensor = self.goal_tensor
        # fall = super().termination_check()
        # dp = goal_tensor[..., :3] - self.root_pos
        # dp[:, self.UP_AXIS] = 0
        # dist = dp.square_().sum(-1).sqrt_()
        # too_far = dist-self.init_dist > 3
        # return torch.logical_or(fall, too_far)
        return super().termination_check()

@torch.jit.script
def observe_iccgan_ee(state_hist: torch.Tensor, seq_len: torch.Tensor,
    target_tensor: torch.Tensor, timer: torch.Tensor,
    sp_upper_bound: float, fps: int
):
    ob = observe_iccgan(state_hist, seq_len)    # state_hist = [ob_horizon, NUM_ENVS, 13 * nlink*13]
    
    root_pos = state_hist[-1, :, :3]            #  (1, NUM_ENVS, disc_obs) root global pos of last frame
    root_orient = state_hist[-1, :, 3:7]

    dp = target_tensor[..., :3] - root_pos
    x = dp[:, 0]
    y = dp[:, 1]
    heading_inv = -heading_zup(root_orient)
    c = torch.cos(heading_inv)      # root_orientation의 x-dir의 각도 (inverse) 
    s = torch.sin(heading_inv)
    x, y = c*x-s*y, s*x+c*y         # [[c -s], [s c]] * [x y]^T (local_dp -> root_orient에서 바라본 dp)

    dist = (x*x + y*y).sqrt_()
    sp = dist.mul(fps/timer)        # speed! ... dist/timer->how many dist we should go per step ... dist*fps/timer -> how much distance we should go in 1 sec

    too_close = dist < 1e-5
    x = torch.where(too_close, x, x/dist)   # x/dist: normalized x
    y = torch.where(too_close, y, y/dist)
    sp.clip_(max=sp_upper_bound)
    dist.div_(3).clip_(max=1.5)

    #! what I added
    rhand_idx = 5
    start_idx = 13 + rhand_idx*13
    rhand_pos = state_hist[-1, :, start_idx:start_idx+3]
    rhand_orient = state_hist[-1, :, start_idx+3:start_idx+7]

    rhand_dp = target_tensor[..., :3] - rhand_pos                       # N x 3

    # calculate root_heading
    UP_AXIS = 2
    origin = root_pos.clone()                                           # N x 3
    origin[..., UP_AXIS] = 0     
    heading = heading_zup(root_orient)                                  # N
    up_dir = torch.zeros_like(origin)                                   # N x 3
    up_dir[..., UP_AXIS] = 1
    
    heading_orient_inv = axang2quat(up_dir, -heading)                   # N x 4

    # change x,y,z into root orient
    rhand_local_dp = rotatepoint(heading_orient_inv, rhand_dp)                                      # N x 3
    local_x, local_y, local_z = rhand_local_dp[:, 0], rhand_local_dp[:, 1], rhand_local_dp[:, 2]    # N

    rhand_dist = (local_x*local_x + local_y*local_y + local_z*local_z).sqrt_()                      # N
    
    #! ADDED lhand reward
    # 1. get lhand_dp (target_pos - left hand current pos)
    lhand_idx = 8
    lstart_idx = 13 + lhand_idx*13
    lhand_pos = state_hist[-1, :, lstart_idx:lstart_idx+3]
    lhand_orient = state_hist[-1, :, lstart_idx+3:lstart_idx+7]

    lhand_dp = target_tensor[..., 3:6] - lhand_pos                      # N x 3
    
    # 2. calculate root_heading -> already done in rhand part 

    # 3. change x,y,z into root orient
    lhand_local_dp = rotatepoint(heading_orient_inv, lhand_dp)                                      # N x 3
    local_l_x, local_l_y, local_l_z = lhand_local_dp[:, 0], lhand_local_dp[:, 1], lhand_local_dp[:, 2]    # N

    lhand_dist = (local_l_x*local_l_x + local_l_y*local_l_y + local_l_z*local_l_z).sqrt_()                      # N
    #! ADDED lhand reward


    #! what I added
    # return torch.cat((ob, x.unsqueeze_(-1), y.unsqueeze_(-1), sp.unsqueeze_(-1), dist.unsqueeze_(-1)), -1)
    return torch.cat((ob, local_x.unsqueeze_(-1), local_y.unsqueeze_(-1), local_z.unsqueeze_(-1), rhand_dist.unsqueeze_(-1), 
                    local_l_x.unsqueeze_(-1), local_l_y.unsqueeze_(-1), local_l_z.unsqueeze_(-1), lhand_dist.unsqueeze_(-1)), -1)



class ICCGANHumanoidEE_ref(ICCGANHumanoidEE):  
    RANDOM_INIT = True
    EE_SIZE = 2
    def step(self, actions):
        # goal visualize
        self._motion_sync()
        env_ids = list(range(len(self.envs)))
        self.reset_goal(env_ids)

        # check overtime of goal_motion_time
        if self.viewer is not None: 
            up_over_env_ids, up_in_env_ids, up_key_links, l_over_env_ids, l_in_env_ids, l_key_links = self.goal_motion_overtime_check()
            self.set_char_color([0.0, 0.0, 0.0], up_over_env_ids, up_key_links)
            self.set_char_color((1, 1, 0.5), up_in_env_ids, up_key_links)
            self.set_char_color([0.0, 0.0, 0.0], l_over_env_ids, l_key_links)
            self.set_char_color((1, 0.5, 1), l_in_env_ids, l_key_links)
            
        obs, rews, dones, info = super().step(actions)
        return obs, rews, dones, info
    
    def goal_motion_overtime_check(self):
        # get motion length from goal_motion_ids
        for ref_motion, replay_speed, ob_horizon, discs in self.disc_ref_motion:
            if "upper" in discs[0].name or "full" in discs[0].name:
                up_motion_length = ref_motion.motion_length
                up_key_links = discs[0].key_links
            if "left" in discs[0].name:
                l_motion_length = ref_motion.motion_length                
                l_key_links = discs[0].key_links
        # goal_motion_times
        up_over_env_ids = torch.nonzero(self.goal_motion_times[..., 0] > up_motion_length[0])
        l_over_env_ids = torch.nonzero(self.goal_motion_times[..., 1] > l_motion_length[0])
        up_in_env_ids = torch.nonzero(self.goal_motion_times[..., 0] <= up_motion_length[0])
        l_in_env_ids = torch.nonzero(self.goal_motion_times[..., 1] <= l_motion_length[0])
        return up_over_env_ids, up_in_env_ids, up_key_links, l_over_env_ids, l_in_env_ids, l_key_links

    def create_motion_info(self):
        # self.goal_root_tensor = torch.zeros_like(self.root_tensor, dtype=torch.float32, device=self.device) # [n_envs, 1, 13]
        self.motion_times = torch.zeros(len(self.envs), dtype=torch.float32, device=self.device)

        self.goal_root_tensor = torch.zeros_like(self.root_tensor.repeat(1, self.EE_SIZE, 1), dtype=torch.float32, device=self.device) # [n_envs, 2, 13]
        self.goal_link_tensor = torch.zeros_like(self.link_tensor, dtype=torch.float32, device=self.device)
        self.goal_joint_tensor = torch.zeros_like(self.joint_tensor, dtype=torch.float32, device=self.device)

        self.goal_motion_ids = torch.zeros([len(self.envs), self.EE_SIZE], dtype=torch.int32, device=self.device)
        self.goal_motion_times = torch.zeros([len(self.envs), self.EE_SIZE], dtype=torch.float32, device=self.device)

        self.offset_time = torch.zeros([len(self.envs), self.EE_SIZE], dtype=torch.float32, device=self.device)
        self.etime = torch.zeros([len(self.envs), self.EE_SIZE], dtype=torch.float32, device=self.device)

    def create_tensors(self):
        super().create_tensors()
        self.create_motion_info()
        n_links = self.gym.get_actor_rigid_body_count(self.envs[0], 0)
        n_dofs = self.gym.get_actor_dof_count(self.envs[0], 0)
        #reference link tensors and joint tensors
        self.up_goal_root_pos, self.up_goal_root_orient = self.goal_root_tensor[:, 0, :3], self.goal_root_tensor[:, 0, 3:7]
        self.l_goal_root_pos, self.l_goal_root_orient = self.goal_root_tensor[:, 1, :3], self.goal_root_tensor[:, 1, 3:7]
        if self.goal_link_tensor.size(1) > n_links:
            self.goal_link_pos, self.goal_link_orient = self.goal_link_tensor[:, :n_links, :3], self.goal_link_tensor[:, :n_links, 3:7]
            self.goal_link_lin_vel, self.goal_link_ang_vel = self.goal_link_tensor[:, :n_links, 7:10], self.goal_link_tensor[:, :n_links, 10:13]
            self.goal_char_link_tensor = self.goal_link_tensor[:, :n_links]
        else:
            self.goal_link_pos, self.goal_link_orient = self.goal_link_tensor[..., :3], self.goal_link_tensor[..., 3:7]
            self.goal_link_lin_vel, self.goal_link_ang_vel = self.goal_link_tensor[..., 7:10], self.goal_link_tensor[..., 10:13]
            self.goal_char_goal_link_tensor = self.goal_link_tensor
        if self.goal_joint_tensor.size(1) > n_dofs:
            self.goal_joint_pos, self.goal_joint_vel = self.goal_joint_tensor[:, :n_dofs, 0], self.goal_joint_tensor[:, :n_dofs, 1]
            self.goal_char_joint_tensor = self.goal_joint_tensor[:, :n_dofs]
        else:
            self.goal_joint_pos, self.goal_joint_vel = self.goal_joint_tensor[..., 0], self.goal_joint_tensor[..., 1]
            self.goal_char_joint_tensor = self.goal_joint_tensor


    def init_state(self, env_ids):
        motion_ids, motion_times = self.ref_motion.sample(len(env_ids))

        self.motion_times[env_ids] = torch.tensor(motion_times, dtype=torch.float32, device=self.device)

        # initialize goal_motion
        for ref_motion, _, _, discs in self.disc_ref_motion:
            if "upper" in discs[0].name or "full" in discs[0].name:
                up_goal_motion_ids, _, up_goal_motion_etime = ref_motion.generate_motion_patch(len(env_ids))
                up_goal_offset_time = ref_motion.randomize_offset(len(env_ids))               # randomize_offset to init_env

            if "left" in discs[0].name:
                l_goal_motion_ids, _, l_goal_motion_etime = ref_motion.generate_motion_patch(len(env_ids))
                l_goal_offset_time = ref_motion.randomize_offset(len(env_ids))               # randomize_offset to init_env
            else: 
                pass

        self.goal_motion_ids[env_ids, 0] = torch.tensor(up_goal_motion_ids, dtype=torch.int32, device=self.device)
        self.goal_motion_ids[env_ids, 1] = torch.tensor(l_goal_motion_ids, dtype=torch.int32, device=self.device)
        
        self.etime[env_ids, 0] = torch.tensor(up_goal_motion_etime, dtype=torch.float32, device=self.device)
        self.etime[env_ids, 1] = torch.tensor(l_goal_motion_etime, dtype=torch.float32, device=self.device)

        self.offset_time[env_ids, 0] = torch.tensor(up_goal_offset_time, dtype=torch.float32, device=self.device)
        self.offset_time[env_ids, 1] = torch.tensor(l_goal_offset_time, dtype=torch.float32, device=self.device)

        # print("\n---------------INIT STATE: {}---------------\n".format(env_ids))
        
        if self.viewer is not None: 
            self.set_char_color([1, 1, 1], env_ids)

        return self.ref_motion.state(motion_ids, motion_times)

    def _observe(self, env_ids):
        if env_ids is None:
            return observe_iccgan_ee(
                self.state_hist[-self.ob_horizon:], self.ob_seq_lens,
                self.goal_tensor, self.goal_timer, sp_upper_bound=self.sp_upper_bound, fps=self.fps
            )
        else:
            return observe_iccgan_ee(
                self.state_hist[-self.ob_horizon:][:, env_ids], self.ob_seq_lens[env_ids],
                self.goal_tensor[env_ids], self.goal_timer[env_ids], sp_upper_bound=self.sp_upper_bound, fps=self.fps
            )
        
    def update_viewer(self):
        super().update_viewer()
        # self.gym.clear_lines(self.viewer)
        # self.visualize_ee_positions()
        self.visualize_goal_positions()
        # self.visualize_origin()
        # self.visualize_ego_ee()
    
    
    def _motion_sync(self):
        def _get_dt_dttensor(device, dt, n_inst, replay_speed=None):
            if replay_speed is not None:
                dt /= replay_speed(n_inst)
                dt_tensor = torch.tensor(dt, dtype=torch.float32, device=device)                  # list -> tensor
            else:
                dt_tensor = torch.tensor(dt, dtype=torch.float32, device=device).repeat(n_inst)   # float -> tensor
            return dt, dt_tensor
        
        def getIntersection_Method1(a, b):
            a = a.detach().cpu().numpy()
            b = b.detach().cpu().numpy()
            intersection = np.intersect1d(a, b)
            return torch.from_numpy(intersection)
        
        def get_env_ids_infos(lifetime, offset_time):
            not_init_env_ids = torch.nonzero(lifetime).view(-1)
            init_env_ids = (lifetime == 0).nonzero().view(-1)
            
            still_motion = torch.nonzero(offset_time).view(-1)
            move_motion = (offset_time == 0).nonzero().view(-1)
            return not_init_env_ids, init_env_ids, still_motion, move_motion
        
        n_inst = len(self.envs)
        env_ids = list(range(n_inst))

        root_tensor = torch.zeros_like(self.goal_root_tensor[env_ids])  # [N_ENV, 2, 13]
        link_tensor = torch.zeros_like(self.goal_link_tensor[env_ids])  # [N_ENV, N_LINK, 13]
        joint_tensor = torch.zeros_like(self.goal_joint_tensor[env_ids])# [N_ENV, N_DOF, 13]

        for ref_motion, replay_speed, ob_horizon, discs in self.disc_ref_motion:
            key_links = discs[0].key_links
            if "upper" in discs[0].name or "full" in discs[0].name:
                dt, dt_tensor = _get_dt_dttensor(self.device, self.step_time, n_inst, replay_speed)
                motion_ids, _ = ref_motion.sample(n_inst, truncate_time=dt*(ob_horizon-1))
                not_init_env_ids, init_env_ids, still_motion, move_motion = get_env_ids_infos(self.lifetime, self.offset_time[:, 0])
                # sill motion
                if len(still_motion):
                    # 1 씩 감소
                    self.offset_time[still_motion, 0] -= 1
                    torch.clamp(self.offset_time[:, 0], min=0) 
                    # motion in still 그대로!
                    print("self.goal_motion_times[still_motion, 0]: ", self.goal_motion_times[still_motion, 0])
                    self.goal_motion_times[still_motion, 0] = self.goal_motion_times[still_motion, 0]
                # motion starts!
                if len(move_motion):
                    move = getIntersection_Method1(not_init_env_ids, move_motion)
                    not_move = getIntersection_Method1(init_env_ids, move_motion)

                    # init 안된 친구는 dt를 계속 더해줌
                    if (len(move)):
                        # print("move!! WITHOUT INIT: ", self.offset_time.item(), self.goal_motion_times.item(), self.etime.item())
                        self.goal_motion_times[move, 0] = self.goal_motion_times[move, 0] + dt_tensor[move.cpu()]
                    # init 된 친구는 0                        
                    if (len(not_move)):
                        # print("INIT")
                        self.goal_motion_times[not_move, 0] = self.goal_motion_times[not_move, 0] + torch.zeros(len(not_move), dtype=torch.float32, device=self.device)

                # self.motion_times = self.motion_times + dt_tensor
                motion_times0 = self.goal_motion_times[:, 0].cpu().numpy()
                up_root_tensor, up_link_tensor, up_joint_tensor = ref_motion.state(motion_ids, motion_times0)

                root_tensor[env_ids, 0] = up_root_tensor
                link_tensor[..., key_links, :] = up_link_tensor[..., key_links, :]
                for idx in key_links:
                    joint_tensor[..., self.DOF_OFFSET[idx]:self.DOF_OFFSET[idx+1], :] = \
                    up_joint_tensor[..., self.DOF_OFFSET[idx]:self.DOF_OFFSET[idx+1], :]

                # self.goal_motion_times가 etime을 over 했을 때 
                over_etime = torch.nonzero(self.goal_motion_times[:, 0] - self.etime[:, 0] > 0.001).view(-1)
                if len(over_etime):
                    # over_time이면서 offset_time이 0 이하여야
                    still_motion = (self.offset_time[:, 0] == 0).nonzero().view(-1)
                    reset = getIntersection_Method1(over_etime, still_motion)

                    # print("reset: ", reset)
                    if len(reset):
                        goal_motion_ids, goal_motion_stime, goal_motion_etime = ref_motion.generate_motion_patch(len(reset))
                        goal_offset_time = ref_motion.randomize_offset(len(reset))               # randomize_offset to init_env

                        self.goal_motion_ids[reset, 0] = torch.tensor(goal_motion_ids, dtype=torch.int32, device=self.device)
                        self.goal_motion_times[reset, 0] = torch.tensor(goal_motion_stime, dtype=torch.float32, device=self.device)
                        
                        self.etime[reset, 0] = torch.tensor(goal_motion_etime, dtype=torch.float32, device=self.device) 
                        self.offset_time[reset, 0] = torch.tensor(goal_offset_time, dtype=torch.float32, device=self.device)

            elif "left" in discs[0].name:
                dt, dt_tensor = _get_dt_dttensor(self.device, self.step_time, n_inst, replay_speed)
                motion_ids, _ = ref_motion.sample(n_inst, truncate_time=dt*(ob_horizon-1))
                not_init_env_ids, init_env_ids, still_motion, move_motion = get_env_ids_infos(self.lifetime, self.offset_time[:, 0])
                # sill motion
                if len(still_motion):
                    # 1 씩 감소
                    self.offset_time[still_motion, 1] -= 1
                    torch.clamp(self.offset_time[:, 1], min=0) 
                    # motion in still 그대로!
                    self.goal_motion_times[still_motion, 1] = self.goal_motion_times[still_motion, 1]
                # motion starts!
                if len(move_motion):
                    move = getIntersection_Method1(not_init_env_ids, move_motion)
                    not_move = getIntersection_Method1(init_env_ids, move_motion)

                    # init 안된 친구는 dt를 계속 더해줌
                    if (len(move)):
                        # print("move!! WITHOUT INIT: ", self.offset_time.item(), self.goal_motion_times.item(), self.etime.item())
                        self.goal_motion_times[move, 1] = self.goal_motion_times[move, 1] + dt_tensor[move.cpu()]
                    # init 된 친구는 0                        
                    if (len(not_move)):
                        # print("INIT")
                        self.goal_motion_times[not_move, 1] = self.goal_motion_times[not_move, 1] + torch.zeros(len(not_move), dtype=torch.float32, device=self.device)

                # self.motion_times = self.motion_times + dt_tensor
                motion_times0 = self.goal_motion_times[:, 1].cpu().numpy()

                l_root_tensor, l_link_tensor, l_joint_tensor = ref_motion.state(motion_ids, motion_times0)

                root_tensor[env_ids, 1] = l_root_tensor
                link_tensor[..., key_links, :] = l_link_tensor[..., key_links, :]
                for idx in key_links:
                    joint_tensor[..., self.DOF_OFFSET[idx]:self.DOF_OFFSET[idx+1], :] = \
                    l_joint_tensor[..., self.DOF_OFFSET[idx]:self.DOF_OFFSET[idx+1], :]

                # self.goal_motion_times가 etime을 over 했을 때 
                over_etime = torch.nonzero(self.goal_motion_times[:, 1] - self.etime[:, 1] > 0.001).view(-1)
                if len(over_etime):
                    # over_time이면서 offset_time이 0 이하여야
                    still_motion = (self.offset_time[:, 1] == 0).nonzero().view(-1)
                    reset = getIntersection_Method1(over_etime, still_motion)

                    # print("reset: ", reset)
                    if len(reset):
                        goal_motion_ids, goal_motion_stime, goal_motion_etime = ref_motion.generate_motion_patch(len(reset))
                        goal_offset_time = ref_motion.randomize_offset(len(reset))               # randomize_offset to init_env

                        self.goal_motion_ids[reset, 1] = torch.tensor(goal_motion_ids, dtype=torch.int32, device=self.device)
                        self.goal_motion_times[reset, 1] = torch.tensor(goal_motion_stime, dtype=torch.float32, device=self.device)
                        
                        self.etime[reset, 1] = torch.tensor(goal_motion_etime, dtype=torch.float32, device=self.device) 
                        self.offset_time[reset, 1] = torch.tensor(goal_offset_time, dtype=torch.float32, device=self.device)
            
            else:
                dt, dt_tensor = _get_dt_dttensor(self.device, self.step_time, n_inst, replay_speed)
                # humanoid 시간 따로
                self.motion_times = self.motion_times + dt_tensor
                
                motion_ids, motion_times = ref_motion.sample(n_inst, truncate_time=dt*(ob_horizon-1))
                _, other_link_tensor, other_joint_tensor = ref_motion.state(motion_ids, self.motion_times.cpu().numpy())
                link_tensor[..., key_links, :] = other_link_tensor[..., key_links, :]
                for idx in key_links:
                    joint_tensor[..., self.DOF_OFFSET[idx]:self.DOF_OFFSET[idx+1], :] = \
                    other_joint_tensor[..., self.DOF_OFFSET[idx]:self.DOF_OFFSET[idx+1], :]

        self.goal_root_tensor[env_ids] = root_tensor
        self.goal_link_tensor[env_ids] = link_tensor
        self.goal_joint_tensor[env_ids] = joint_tensor


    def reset_envs(self, env_ids):
        ref_root_tensor, ref_link_tensor, ref_joint_tensor = self.init_state(env_ids)

        self.root_tensor[env_ids] = ref_root_tensor
        self.link_tensor[env_ids] = ref_link_tensor
        self.joint_tensor[env_ids] = ref_joint_tensor

        actor_ids = self.actor_ids[env_ids].flatten()
        n_actor_ids = len(actor_ids)
        actor_ids = gymtorch.unwrap_tensor(actor_ids)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
            gymtorch.unwrap_tensor(self.root_tensor),
            actor_ids, n_actor_ids
        )
        actor_ids = self.actor_ids_having_dofs[env_ids].flatten()
        n_actor_ids = len(actor_ids)
        actor_ids = gymtorch.unwrap_tensor(actor_ids)
        self.gym.set_dof_state_tensor_indexed(self.sim,
            gymtorch.unwrap_tensor(self.joint_tensor),
            actor_ids, n_actor_ids
        )

        self.lifetime[env_ids] = 0
        #! b/c reset goal in every steps!
        # self.reset_goal(env_ids)
            
    def reset_goal(self, env_ids, goal_tensor=None, goal_timer=None):
        #! shallow copy: 이렇게 되면 goal_tensor가 바뀌면 self.goal_tensor도 바뀐다!
        if goal_tensor is None: goal_tensor = self.goal_tensor
        if goal_timer is None: goal_timer = self.goal_timer
        
        # reset하는 env 개수
        n_envs = len(env_ids)
        ee_links = [5, 8]  # right hand, left hand
        ee_pos = self.goal_link_pos[:, ee_links, :] # [n_envs, n_ee_link, 3]

        all_envs = n_envs == len(self.envs)
        def global_to_ego(root_pos, root_orient, ee_pos, up_axis): # root_pos, root_orient = [n_envs, 3], [n_envs, 4]
            UP_AXIS = up_axis
            origin = root_pos.clone()
            origin[..., UP_AXIS] = 0
            heading = heading_zup(root_orient)                                      # N
            up_dir = torch.zeros_like(origin)                                       # N x 1 x 3
            up_dir[..., UP_AXIS] = 1
            heading_orient_inv = axang2quat(up_dir, -heading)                       # N x 4
            
            if len(ee_pos.size()) == 3:
                heading_orient_inv = heading_orient_inv.unsqueeze(-2).repeat(1, ee_pos.size(-2),1)
                ego_ee_pos = ee_pos - origin.unsqueeze(-2)                             # [n_envs, n_ee_link, 3]
            elif len(ee_pos.size()) == 2:
                ego_ee_pos = ee_pos - origin                                           # [n_envs, 3]
            # change x,y,z into root orient
            ego_ee_pos = rotatepoint(heading_orient_inv, ego_ee_pos)       # [512, n_ee_link, 3] or [n_envs, 3]
            return ego_ee_pos
        
        # ego-centric (ee_pos - origin)
        rhand_ego_ee_pos = global_to_ego(self.up_goal_root_pos[env_ids], self.up_goal_root_orient[env_ids], ee_pos[:, 0, :], 2)
        lhand_ego_ee_pos = global_to_ego(self.l_goal_root_pos[env_ids], self.l_goal_root_orient[env_ids], ee_pos[:, 1, :], 2)

        #! rhand position
        if n_envs == len(self.envs):
            goal_tensor[:, 0] = rhand_ego_ee_pos[:, 0]
            goal_tensor[:, 1] = rhand_ego_ee_pos[:, 1]
            goal_tensor[:, 2] = rhand_ego_ee_pos[:, 2]
        else:
            goal_tensor[env_ids, 0] = rhand_ego_ee_pos[:, 0]
            goal_tensor[env_ids, 1] = rhand_ego_ee_pos[:, 1]
            goal_tensor[env_ids, 2] = rhand_ego_ee_pos[:, 2]
        #!

        #! ADDED lhand position
        if n_envs == len(self.envs):
            goal_tensor[:, 0+3] = lhand_ego_ee_pos[:, 0]
            goal_tensor[:, 1+3] = lhand_ego_ee_pos[:, 1]
            goal_tensor[:, 2+3] = lhand_ego_ee_pos[:, 2]
        else:
            goal_tensor[env_ids, 0+3] = lhand_ego_ee_pos[:, 0]
            goal_tensor[env_ids, 1+3] = lhand_ego_ee_pos[:, 1]
            goal_tensor[env_ids, 2+3] = lhand_ego_ee_pos[:, 2]
        #!

    def reward(self, goal_tensor=None, goal_timer=None):
        if goal_tensor is None: goal_tensor = self.goal_tensor
        if goal_timer is None: goal_timer = self.goal_timer

        p = self.root_pos                                       # 현재 root_pos
        p_ = self.state_hist[-1][:, :3]                         # 이전 root_pos (goal_tensor 구했을 때의 root_pos부터 시작!  / action apply 되기 이전)

        dp_ = goal_tensor[..., :3] - p_                                  # root_pos에서 target 지점까지의 global (dx, dy)
        dp_[:, self.UP_AXIS] = 0
        dist_ = torch.linalg.norm(dp_, ord=2, dim=-1)
        v_ = dp_.div_(goal_timer.unsqueeze(-1)*self.step_time)  # v_: desired veloicty (total distance / sec)

        v_mag = torch.linalg.norm(v_, ord=2, dim=-1)
        sp_ = (dist_/self.step_time).clip_(max=v_mag.clip(min=self.sp_lower_bound, max=self.sp_upper_bound))
        v_ *= (sp_/v_mag).unsqueeze_(-1)                       # desired velocity

        dp = p - p_                                            # (현재 root - 이전 root)
        dp[:, self.UP_AXIS] = 0
        v = dp / self.step_time                                # current velocity: dp / duration 
        r = (v - v_).square_().sum(1).mul_(-3/(sp_*sp_)).exp_()

        dp = goal_tensor[..., :3] - p
        dp[:, self.UP_AXIS] = 0
        dist = torch.linalg.norm(dp, ord=2, dim=-1)
        self.near = dist < self.goal_radius

        r[self.near] = 1
        
        if self.viewer is not None:
            self.goal_timer[self.near] = self.goal_timer[self.near].clip(max=20)
        
        rarm_len, larm_len = self.get_link_len([2,3,4], [3,4,5]), self.get_link_len([2,6,7], [6,7,8])
        rarm_len, larm_len = rarm_len.sum(dim=0), larm_len.sum(dim=0)
        rarm_len, larm_len = rarm_len.repeat(len(self.envs)), larm_len.repeat(len(self.envs))

        #! ego-centric (ee_pos - origin)
        UP_AXIS = 2
        root_pos, root_orient = self.root_pos, self.root_orient                 # [n_envs, 3], [n_envs, 4]
        origin = root_pos.clone()
        origin[..., UP_AXIS] = 0
        heading = heading_zup(root_orient)                                      # N
        up_dir = torch.zeros_like(origin)                                       # N x 1 x 3
        up_dir[..., UP_AXIS] = 1
        heading_orient_inv = axang2quat(up_dir, -heading)                       # N x 4

        #! ADDED rhand reward
        target_ego_rhand_pos = goal_tensor[..., :3]
        rhand_pos = self.link_pos[:, self.r_hand_link] + rotatepoint(ee_rot[:, 0], self.r_lpos.to(self.device)) 
        ego_rhand_pos = rhand_pos - origin
        ego_rhand_pos = rotatepoint(heading_orient_inv, ego_rhand_pos)          # N x 3

        e = torch.linalg.norm(target_ego_rhand_pos.sub(ego_rhand_pos), ord=2, dim=-1).div_(rarm_len)
        rhand_rew = e.mul_(-2).exp_()
        #! ADDED rhand reward

        #! ADDED lhand reward
        target_ego_lhand_pos = goal_tensor[..., 3:6]
        lhand_pos = self.link_pos[:, self.l_hand_link]
        ego_lhand_pos = lhand_pos - origin
        ego_lhand_pos = rotatepoint(heading_orient_inv, ego_lhand_pos)          # N x 3

        l_e = torch.linalg.norm(target_ego_lhand_pos.sub(ego_lhand_pos), ord=2, dim=-1).div_(larm_len)
        lhand_rew = l_e.mul_(-2).exp_()
        #! ADDED lhand reward

        total_r = (0.5 * rhand_rew + 0.5 * lhand_rew)        
        return total_r.unsqueeze_(-1)
    
    def visualize_origin(self):
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, 1.0, 0.0)
        pose.r = gymapi.Quat(0, 0.0, 0.0, 1)
        for i in range(len(self.envs)):
            axes_geom = gymutil.AxesGeometry(1)
            sphere_geom = gymutil.WireframeSphereGeometry(0.04, 16, 16, None, color=(0, 0, 0))   # pink

            gymutil.draw_lines(axes_geom, self.gym, self.viewer, self.envs[i], pose)
            gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], pose)   

        pass

    def visualize_ee_positions(self):
        ee_links = [2, 5, 8, 0]
        hsphere_geom = gymutil.WireframeSphereGeometry(0.04, 16, 16, None, color=(1, 1, 1))     # white
        lsphere_geom = gymutil.WireframeSphereGeometry(0.04, 16, 16, None, color=(1, 0.3, 1))   # pink
        rsphere_geom = gymutil.WireframeSphereGeometry(0.04, 16, 16, None, color=(1, 1, 0.3))   # yellow
        rootsphere_geom = gymutil.WireframeSphereGeometry(0.04, 16, 16, None, color=(0.3, 1, 1))   # pink
        ee_pos = self.goal_link_pos[:, ee_links, :]
        ee_pos = ee_pos        
        for i in range(len(self.envs)):
            head_pos = ee_pos[i, 0]
            rhand_pos = ee_pos[i, 1]
            lhand_pos = ee_pos[i, 2]
            root_pos = ee_pos[i, 3]
            head_pose = gymapi.Transform(gymapi.Vec3(head_pos[0], head_pos[1], head_pos[2]), r=None)
            rhand_pose = gymapi.Transform(gymapi.Vec3(rhand_pos[0], rhand_pos[1], rhand_pos[2]), r=None)
            lhand_pose = gymapi.Transform(gymapi.Vec3(lhand_pos[0], lhand_pos[1], lhand_pos[2]), r=None)
            root_pose = gymapi.Transform(gymapi.Vec3(root_pos[0], root_pos[1], root_pos[2]), r=None)
            gymutil.draw_lines(hsphere_geom, self.gym, self.viewer, self.envs[i], head_pose)    # white 
            gymutil.draw_lines(lsphere_geom, self.gym, self.viewer, self.envs[i], lhand_pose)   # pink
            gymutil.draw_lines(rsphere_geom, self.gym, self.viewer, self.envs[i], rhand_pose)
            gymutil.draw_lines(rootsphere_geom, self.gym, self.viewer, self.envs[i], root_pose)   

    def visualize_goal_positions(self):
        up_ee_links = [3, 4, 5]
        l_key_links = [6, 7, 8]
        lower_key_links = [0, 1, 2, 9, 10, 11, 12, 13, 14]

        lsphere_geom = gymutil.WireframeSphereGeometry(0.04, 16, 16, None, color=(1, 0.3, 1))       # pink
        rsphere_geom = gymutil.WireframeSphereGeometry(0.04, 16, 16, None, color=(1, 1, 0.3))       # yellow
        rootsphere_geom = gymutil.WireframeSphereGeometry(0.04, 16, 16, None, color=(0.3, 1, 1))    # pink

        ee_pos = self.goal_link_pos
        for i in range(len(self.envs)):
            for link in up_ee_links:
                link_pos = ee_pos[i, link]
                link_pose = gymapi.Transform(gymapi.Vec3(link_pos[0], link_pos[1], link_pos[2]), r=None)
                gymutil.draw_lines(rsphere_geom, self.gym, self.viewer, self.envs[i], link_pose)    # white 
            for link in l_key_links:
                link_pos = ee_pos[i, link]
                link_pose = gymapi.Transform(gymapi.Vec3(link_pos[0], link_pos[1], link_pos[2]), r=None)
                gymutil.draw_lines(lsphere_geom, self.gym, self.viewer, self.envs[i], link_pose)    # white
            for link in lower_key_links:
                link_pos = ee_pos[i, link]
                link_pose = gymapi.Transform(gymapi.Vec3(link_pos[0], link_pos[1], link_pos[2]), r=None)
                gymutil.draw_lines(rootsphere_geom, self.gym, self.viewer, self.envs[i], link_pose)    # white

    def visualize_ego_ee(self):
        lsphere_geom = gymutil.WireframeSphereGeometry(0.04, 16, 16, None, color=(1, 1, 1))       # pink
        rsphere_geom = gymutil.WireframeSphereGeometry(0.04, 16, 16, None, color=(1, 1, 1))       # yellow
        rootsphere_geom = gymutil.WireframeSphereGeometry(0.04, 16, 16, None, color=(0.3, 1, 1))    # pink
        ee_rpos = self.goal_tensor[:, 0:3]
        ee_lpos = self.goal_tensor[:, 3:6]
        root_pos = torch.zeros_like(self.root_pos)
        root_pos[:, 0], root_pos[:, 1] = self.root_pos[:, 0], self.root_pos[:, 1]
        for i in range(len(self.envs)):
            rhand_pos = ee_rpos + root_pos
            lhand_pos = ee_lpos + root_pos
            rhand_pose = gymapi.Transform(gymapi.Vec3(rhand_pos[:, 0], rhand_pos[:, 1], rhand_pos[:, 2]), r=None)
            lhand_pose = gymapi.Transform(gymapi.Vec3(lhand_pos[:, 0], lhand_pos[:, 1], lhand_pos[:, 2]), r=None)
            root_pose = gymapi.Transform(gymapi.Vec3(root_pos[:, 0], root_pos[:, 1], root_pos[:, 2]), r=None)
            gymutil.draw_lines(rsphere_geom, self.gym, self.viewer, self.envs[i], rhand_pose)
            gymutil.draw_lines(lsphere_geom, self.gym, self.viewer, self.envs[i], lhand_pose)
            gymutil.draw_lines(rootsphere_geom, self.gym, self.viewer, self.envs[i], root_pose)

class ICCGANHumanoidVR(ICCGANHumanoidEE):  
    RANDOM_INIT = True
    EE_SIZE = 2
    GOAL_TENSOR_DIM = 3 + 3 + 3 + 4                 # global position of right/left hand / head controller target (X, Y, Z) - where they should reach

    RPOS_COEFF = 0.25
    LPOS_COEFF = 0.25
    HPOS_COEFF = 0.25
    HROT_COEFF = 0.25
    
    def __init__(self, *args, 
                 sensor_inputs: Optional[Dict[str, SensorInputConfig]]=None,
                 **kwargs):
        self.rpos_coeff = parse_kwarg(kwargs, "rhand_pos", self.RPOS_COEFF)
        self.lpos_coeff = parse_kwarg(kwargs, "lhand_pos", self.LPOS_COEFF)
        self.hpos_coeff = parse_kwarg(kwargs, "hmd_pos", self.HPOS_COEFF)
        self.hrot_coeff = parse_kwarg(kwargs, "hmd_rot", self.HROT_COEFF)
        self.sensor_inputs = sensor_inputs
        super().__init__(*args, **kwargs)

    def step(self, actions):
        # goal visualize
        self._motion_sync()
        env_ids = list(range(len(self.envs)))
        self.reset_goal(env_ids)

        # check overtime of goal_motion_time
        if self.viewer is not None: 
            up_over_env_ids, up_in_env_ids, up_key_links, l_over_env_ids, l_in_env_ids, l_key_links = self.goal_motion_overtime_check()
            self.set_char_color([0.0, 0.0, 0.0], up_over_env_ids, up_key_links)
            self.set_char_color((1, 1, 0.5), up_in_env_ids, up_key_links)
            self.set_char_color([0.0, 0.0, 0.0], l_over_env_ids, l_key_links)
            self.set_char_color((1, 0.5, 1), l_in_env_ids, l_key_links)
            
        obs, rews, dones, info = super().step(actions)
        return obs, rews, dones, info
    
    def goal_motion_overtime_check(self):
        # get motion length from goal_motion_ids
        for ref_motion, replay_speed, ob_horizon, discs in self.disc_ref_motion:
            if "upper" in discs[0].name or "full" in discs[0].name:
                up_motion_length = ref_motion.motion_length
                up_key_links = discs[0].key_links
            if "left" in discs[0].name:
                l_motion_length = ref_motion.motion_length                
                l_key_links = discs[0].key_links
        # goal_motion_times
        up_over_env_ids = torch.nonzero(self.goal_motion_times[..., 0] > up_motion_length[0])
        up_in_env_ids = torch.nonzero(self.goal_motion_times[..., 0] <= up_motion_length[0])
        if 'l_motion_length' in locals():
            l_over_env_ids = torch.nonzero(self.goal_motion_times[..., 1] > l_motion_length[0])
            l_in_env_ids = torch.nonzero(self.goal_motion_times[..., 1] <= l_motion_length[0])
        else:
            l_over_env_ids, l_in_env_ids, l_key_links = [], [],[]
        return up_over_env_ids, up_in_env_ids, up_key_links, l_over_env_ids, l_in_env_ids, l_key_links

    def create_motion_info(self):
        # self.goal_root_tensor = torch.zeros_like(self.root_tensor, dtype=torch.float32, device=self.device) # [n_envs, 1, 13]
        self.motion_times = torch.zeros(len(self.envs), dtype=torch.float32, device=self.device)

        self.goal_root_tensor = torch.zeros_like(self.root_tensor.repeat(1, self.EE_SIZE, 1), dtype=torch.float32, device=self.device) # [n_envs, 2, 13]
        self.goal_link_tensor = torch.zeros_like(self.link_tensor, dtype=torch.float32, device=self.device)
        self.goal_joint_tensor = torch.zeros_like(self.joint_tensor, dtype=torch.float32, device=self.device)

        self.goal_motion_ids = torch.zeros([len(self.envs), self.EE_SIZE], dtype=torch.int32, device=self.device)
        self.goal_motion_times = torch.zeros([len(self.envs), self.EE_SIZE], dtype=torch.float32, device=self.device)

        self.offset_time = torch.zeros([len(self.envs), self.EE_SIZE], dtype=torch.float32, device=self.device)
        self.etime = torch.zeros([len(self.envs), self.EE_SIZE], dtype=torch.float32, device=self.device)

        # get file path of rlh localPos, Rot, xy_pressed
        for name, sensorconfig in self.sensor_inputs.items():
            print("\n=======\n", name, ": sensor input path well detected", "\n=======\n")
        rlh_localPos = np.load(os.getcwd() + sensorconfig.rlh_localPos)
        rlh_localpos = torch.tensor(rlh_localPos, dtype=torch.float32, device=self.device)
        
        r_localpos, l_localpos, h_localpos = rlh_localpos[..., 0:3], rlh_localpos[..., 3:6], rlh_localpos[..., 6:9]
        #! Should this not be averaged?? 
        self.r_lpos, self.l_lpos, self.h_lpos = torch.mean(r_localpos, dim=0), torch.mean(l_localpos, dim=0), torch.mean(h_localpos, dim=0) #(3, )
        
        rlh_localRot = np.load(os.getcwd() + sensorconfig.rlh_localRot)
        rlh_localRot = torch.tensor(rlh_localRot, dtype=torch.float32, device=self.device)
        self.h_lrot = rlh_localRot[..., 8:12]
        # self.r_lrot, self.l_lrot, self.h_lrot = torch.mean(r_localRot, dim=0), torch.mean(l_localRot, dim=0), torch.mean(h_localRot, dim=0) #(3, )
       
        #! should add variables for x,y pressed!

    def create_tensors(self):
        super().create_tensors()
        self.create_motion_info()
        n_links = self.gym.get_actor_rigid_body_count(self.envs[0], 0)
        n_dofs = self.gym.get_actor_dof_count(self.envs[0], 0)
        #reference link tensors and joint tensors
        self.up_goal_root_pos, self.up_goal_root_orient = self.goal_root_tensor[:, 0, :3], self.goal_root_tensor[:, 0, 3:7]
        self.l_goal_root_pos, self.l_goal_root_orient = self.goal_root_tensor[:, 1, :3], self.goal_root_tensor[:, 1, 3:7]
        if self.goal_link_tensor.size(1) > n_links:
            self.goal_link_pos, self.goal_link_orient = self.goal_link_tensor[:, :n_links, :3], self.goal_link_tensor[:, :n_links, 3:7]
            self.goal_link_lin_vel, self.goal_link_ang_vel = self.goal_link_tensor[:, :n_links, 7:10], self.goal_link_tensor[:, :n_links, 10:13]
            self.goal_char_link_tensor = self.goal_link_tensor[:, :n_links]
        else:
            self.goal_link_pos, self.goal_link_orient = self.goal_link_tensor[..., :3], self.goal_link_tensor[..., 3:7]
            self.goal_link_lin_vel, self.goal_link_ang_vel = self.goal_link_tensor[..., 7:10], self.goal_link_tensor[..., 10:13]
            self.goal_char_goal_link_tensor = self.goal_link_tensor
        if self.goal_joint_tensor.size(1) > n_dofs:
            self.goal_joint_pos, self.goal_joint_vel = self.goal_joint_tensor[:, :n_dofs, 0], self.goal_joint_tensor[:, :n_dofs, 1]
            self.goal_char_joint_tensor = self.goal_joint_tensor[:, :n_dofs]
        else:
            self.goal_joint_pos, self.goal_joint_vel = self.goal_joint_tensor[..., 0], self.goal_joint_tensor[..., 1]
            self.goal_char_joint_tensor = self.goal_joint_tensor


    def init_state(self, env_ids):
        motion_ids, motion_times = self.ref_motion.sample(len(env_ids))

        self.motion_times[env_ids] = torch.tensor(motion_times, dtype=torch.float32, device=self.device)

        # initialize goal_motion
        l_goal_motion_ids = None
        for ref_motion, _, _, discs in self.disc_ref_motion:
            if "upper" in discs[0].name or "full" in discs[0].name:
                up_goal_motion_ids, _, up_goal_motion_etime = ref_motion.generate_motion_patch(len(env_ids))
                up_goal_offset_time = ref_motion.randomize_offset(len(env_ids))               # randomize_offset to init_env

            if "left" in discs[0].name:
                l_goal_motion_ids, _, l_goal_motion_etime = ref_motion.generate_motion_patch(len(env_ids))
                l_goal_offset_time = ref_motion.randomize_offset(len(env_ids))               # randomize_offset to init_env
            else:
                pass

        self.goal_motion_ids[env_ids, 0] = torch.tensor(up_goal_motion_ids, dtype=torch.int32, device=self.device)
        self.etime[env_ids, 0] = torch.tensor(up_goal_motion_etime, dtype=torch.float32, device=self.device)
        self.offset_time[env_ids, 0] = torch.tensor(up_goal_offset_time, dtype=torch.float32, device=self.device)

        if (l_goal_motion_ids != None):
            self.goal_motion_ids[env_ids, 1] = torch.tensor(l_goal_motion_ids, dtype=torch.int32, device=self.device)
            self.etime[env_ids, 1] = torch.tensor(l_goal_motion_etime, dtype=torch.float32, device=self.device)
            self.offset_time[env_ids, 1] = torch.tensor(l_goal_offset_time, dtype=torch.float32, device=self.device)

        # print("\n---------------INIT STATE: {}---------------\n".format(env_ids))
        
        if self.viewer is not None: 
            self.set_char_color([1, 1, 1], env_ids)

        return self.ref_motion.state(motion_ids, motion_times)

    def _observe(self, env_ids):
        if env_ids is None:
            return observe_iccgan_ee(
                self.state_hist[-self.ob_horizon:], self.ob_seq_lens,
                self.goal_tensor, self.goal_timer, sp_upper_bound=self.sp_upper_bound, fps=self.fps
            )
        else:
            return observe_iccgan_ee(
                self.state_hist[-self.ob_horizon:][:, env_ids], self.ob_seq_lens[env_ids],
                self.goal_tensor[env_ids], self.goal_timer[env_ids], sp_upper_bound=self.sp_upper_bound, fps=self.fps
            )
        
    def update_viewer(self):
        super().update_viewer()
        # self.gym.clear_lines(self.viewer)
        # self.visualize_ee_positions()
        self.visualize_goal_positions()
        self.visualize_control_positions()
        self.visualize_hmd_rotations()
        # self.visualize_origin()
        # self.visualize_ego_ee()
    
    
    def _motion_sync(self):
        def _get_dt_dttensor(device, dt, n_inst, replay_speed=None):
            if replay_speed is not None:
                dt /= replay_speed(n_inst)
                dt_tensor = torch.tensor(dt, dtype=torch.float32, device=device)                  # list -> tensor
            else:
                dt_tensor = torch.tensor(dt, dtype=torch.float32, device=device).repeat(n_inst)   # float -> tensor
            return dt, dt_tensor
        
        def get_env_ids_infos(lifetime):
            not_init_env_ids = torch.nonzero(lifetime).view(-1)
            init_env_ids = (lifetime == 0).nonzero().view(-1)
            return not_init_env_ids, init_env_ids
        
        n_inst = len(self.envs)
        env_ids = list(range(n_inst))

        root_tensor = torch.zeros_like(self.goal_root_tensor[env_ids])  # [N_ENV, 2, 13]
        link_tensor = torch.zeros_like(self.goal_link_tensor[env_ids])  # [N_ENV, N_LINK, 13]
        joint_tensor = torch.zeros_like(self.goal_joint_tensor[env_ids])# [N_ENV, N_DOF, 13]

        for ref_motion, replay_speed, ob_horizon, discs in self.disc_ref_motion:
            key_links = discs[0].key_links
            if "upper" in discs[0].name or "full" in discs[0].name:
                dt, dt_tensor = _get_dt_dttensor(self.device, self.step_time, n_inst, replay_speed)
                motion_ids, _ = ref_motion.sample(n_inst, truncate_time=dt*(ob_horizon-1))
                not_init_env_ids, init_env_ids = get_env_ids_infos(self.lifetime)

                if len(not_init_env_ids):
                    self.goal_motion_times[not_init_env_ids, 0] = self.goal_motion_times[not_init_env_ids, 0] + dt_tensor[not_init_env_ids.cpu()]
                elif len(init_env_ids):
                    self.goal_motion_times[init_env_ids, 0] = self.goal_motion_times[init_env_ids, 0] + torch.zeros(len(init_env_ids), dtype=torch.float32, device=self.device)
            
                motion_times0 = self.goal_motion_times[:, 0].cpu().numpy()
                up_root_tensor, up_link_tensor, up_joint_tensor = ref_motion.state(motion_ids, motion_times0)

                root_tensor[env_ids, 0] = up_root_tensor
                link_tensor[..., key_links, :] = up_link_tensor[..., key_links, :]
                for idx in key_links:
                    joint_tensor[..., self.DOF_OFFSET[idx]:self.DOF_OFFSET[idx+1], :] = \
                    up_joint_tensor[..., self.DOF_OFFSET[idx]:self.DOF_OFFSET[idx+1], :]

                # self.goal_motion_times가 etime을 over 했을 때 
                over_etime = torch.nonzero(self.goal_motion_times[:, 0] - self.etime[:, 0] > 0.001).view(-1)
                if len(over_etime):
                    # over_time이면서 offset_time이 0 이하여야
                    # still_motion = (self.offset_time[:, 1] == 0).nonzero().view(-1)
                    # reset = getIntersection_Method1(over_etime, still_motion)
                    reset = over_etime
                    # print("reset: ", reset)
                    if len(reset):
                        goal_motion_ids, goal_motion_stime, goal_motion_etime = ref_motion.generate_motion_patch(len(reset))
                        goal_offset_time = ref_motion.randomize_offset(len(reset))               # randomize_offset to init_env

                        self.goal_motion_ids[reset, 0] = torch.tensor(goal_motion_ids, dtype=torch.int32, device=self.device)
                        self.goal_motion_times[reset, 0] = torch.tensor(goal_motion_stime, dtype=torch.float32, device=self.device)
                        
                        self.etime[reset, 0] = torch.tensor(goal_motion_etime, dtype=torch.float32, device=self.device) 
                        self.offset_time[reset, 0] = torch.tensor(goal_offset_time, dtype=torch.float32, device=self.device)
            
            else:
                dt, dt_tensor = _get_dt_dttensor(self.device, self.step_time, n_inst, replay_speed)
                # humanoid 시간 따로
                self.motion_times = self.motion_times + dt_tensor
                
                motion_ids, motion_times = ref_motion.sample(n_inst, truncate_time=dt*(ob_horizon-1))
                _, other_link_tensor, other_joint_tensor = ref_motion.state(motion_ids, self.motion_times.cpu().numpy())
                link_tensor[..., key_links, :] = other_link_tensor[..., key_links, :]
                for idx in key_links:
                    joint_tensor[..., self.DOF_OFFSET[idx]:self.DOF_OFFSET[idx+1], :] = \
                    other_joint_tensor[..., self.DOF_OFFSET[idx]:self.DOF_OFFSET[idx+1], :]

        self.goal_root_tensor[env_ids] = root_tensor
        self.goal_link_tensor[env_ids] = link_tensor
        self.goal_joint_tensor[env_ids] = joint_tensor

    def reset_envs(self, env_ids):
        ref_root_tensor, ref_link_tensor, ref_joint_tensor = self.init_state(env_ids)

        self.root_tensor[env_ids] = ref_root_tensor
        self.link_tensor[env_ids] = ref_link_tensor
        self.joint_tensor[env_ids] = ref_joint_tensor

        actor_ids = self.actor_ids[env_ids].flatten()
        n_actor_ids = len(actor_ids)
        actor_ids = gymtorch.unwrap_tensor(actor_ids)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
            gymtorch.unwrap_tensor(self.root_tensor),
            actor_ids, n_actor_ids
        )
        actor_ids = self.actor_ids_having_dofs[env_ids].flatten()
        n_actor_ids = len(actor_ids)
        actor_ids = gymtorch.unwrap_tensor(actor_ids)
        self.gym.set_dof_state_tensor_indexed(self.sim,
            gymtorch.unwrap_tensor(self.joint_tensor),
            actor_ids, n_actor_ids
        )

        self.lifetime[env_ids] = 0
        #! b/c reset goal in every steps!
        # self.reset_goal(env_ids)
            
    def reset_goal(self, env_ids, goal_tensor=None, goal_timer=None):
        #! shallow copy: 이렇게 되면 goal_tensor가 바뀌면 self.goal_tensor도 바뀐다!
        if goal_tensor is None: goal_tensor = self.goal_tensor
        if goal_timer is None: goal_timer = self.goal_timer
        
        # reset하는 env 개수
        n_envs = len(env_ids)
        ee_links = [5, 8, 2]  # right hand, left hand, head
        ee_pos = self.goal_link_pos[:, ee_links, :] # [n_envs, n_ee_link, 3]
        ee_rot = self.goal_link_orient[:, ee_links, :]   # [num_envs, 3, 4]

        all_envs = n_envs == len(self.envs)
        def global_to_ego(root_pos, root_orient, ee_pos, up_axis): # root_pos, root_orient = [n_envs, 3], [n_envs, 4]
            UP_AXIS = up_axis
            origin = root_pos.clone()
            origin[..., UP_AXIS] = 0
            heading = heading_zup(root_orient)                                      # N
            up_dir = torch.zeros_like(origin)                                       # N x 1 x 3
            up_dir[..., UP_AXIS] = 1
            heading_orient_inv = axang2quat(up_dir, -heading)                       # N x 4
            
            if len(ee_pos.size()) == 3:
                heading_orient_inv = heading_orient_inv.unsqueeze(-2).repeat(1, ee_pos.size(-2),1)
                ego_ee_pos = ee_pos - origin.unsqueeze(-2)                             # [n_envs, n_ee_link, 3]
            elif len(ee_pos.size()) == 2:
                ego_ee_pos = ee_pos - origin                                           # [n_envs, 3]
            # change x,y,z into root orient
            ego_ee_pos = rotatepoint(heading_orient_inv, ego_ee_pos)       # [512, n_ee_link, 3] or [n_envs, 3]
            return ego_ee_pos
        
        # ego-centric (ee_pos - origin)
        # rhand_ego_ee_pos = global_to_ego(self.up_goal_root_pos[env_ids], self.up_goal_root_orient[env_ids], ee_pos[:, 0, :], 2)
        # lhand_ego_ee_pos = global_to_ego(self.l_goal_root_pos[env_ids], self.l_goal_root_orient[env_ids], ee_pos[:, 1, :], 2)
        #! control egocentric position
        rcontrol_pos = ee_pos[:, 0, :] + rotatepoint(ee_rot[:, 0], self.r_lpos.to(self.device))                  # [num_envs, 3] + (3, )
        lcontrol_pos = ee_pos[:, 1, :] + rotatepoint(ee_rot[:, 1], self.l_lpos.to(self.device))                  # [num_envs, 3] + (3, )
        hmd_pos = ee_pos[:, 2, :] + rotatepoint(ee_rot[:, 2], self.h_lpos.to(self.device))                       # [num_envs, 3] + (3, )
        

        #! need to change when l_goal_root_pos is used in three discriminators
        rhand_ego_ee_pos = global_to_ego(self.up_goal_root_pos[env_ids], self.up_goal_root_orient[env_ids], rcontrol_pos, 2)
        # lhand_ego_ee_pos = global_to_ego(self.l_goal_root_pos[env_ids], self.l_goal_root_orient[env_ids], lcontrol_pos, 2)
        lhand_ego_ee_pos = global_to_ego(self.up_goal_root_pos[env_ids], self.up_goal_root_orient[env_ids], lcontrol_pos, 2)
        hmd_ego_ee_pos = global_to_ego(self.up_goal_root_pos[env_ids], self.up_goal_root_orient[env_ids], hmd_pos, 2)

        # rotation (#!300 should be changed with respect to the motion clip)
        hmd_lrot = self.h_lrot[self.lifetime % 300]                     # [num_envs, 4]
        hmd_grot = quat_mul(ee_rot[:, 2], hmd_lrot)                     # [num_envs, 4]

        #! rhand position
        if n_envs == len(self.envs):
            goal_tensor[:, 0] = rhand_ego_ee_pos[:, 0]
            goal_tensor[:, 1] = rhand_ego_ee_pos[:, 1]
            goal_tensor[:, 2] = rhand_ego_ee_pos[:, 2]
        else:
            goal_tensor[env_ids, 0] = rhand_ego_ee_pos[:, 0]
            goal_tensor[env_ids, 1] = rhand_ego_ee_pos[:, 1]
            goal_tensor[env_ids, 2] = rhand_ego_ee_pos[:, 2]
        #!

        #! ADDED lhand position
        if n_envs == len(self.envs):
            goal_tensor[:, 0+3] = lhand_ego_ee_pos[:, 0]
            goal_tensor[:, 1+3] = lhand_ego_ee_pos[:, 1]
            goal_tensor[:, 2+3] = lhand_ego_ee_pos[:, 2]
        else:
            goal_tensor[env_ids, 0+3] = lhand_ego_ee_pos[:, 0]
            goal_tensor[env_ids, 1+3] = lhand_ego_ee_pos[:, 1]
            goal_tensor[env_ids, 2+3] = lhand_ego_ee_pos[:, 2]
        #!

        #! ADDED hmd position
        if n_envs == len(self.envs):
            goal_tensor[:, 0+6] = hmd_ego_ee_pos[:, 0]
            goal_tensor[:, 1+6] = hmd_ego_ee_pos[:, 1]
            goal_tensor[:, 2+6] = hmd_ego_ee_pos[:, 2]
        else:
            goal_tensor[env_ids, 0+6] = hmd_ego_ee_pos[:, 0]
            goal_tensor[env_ids, 1+6] = hmd_ego_ee_pos[:, 1]
            goal_tensor[env_ids, 2+6] = hmd_ego_ee_pos[:, 2]
        #!

        #! ADDED hmd rotation
        if n_envs == len(self.envs):
            goal_tensor[:, 0+9] = hmd_grot[:, 0]
            goal_tensor[:, 1+9] = hmd_grot[:, 1]
            goal_tensor[:, 2+9] = hmd_grot[:, 2]
            goal_tensor[:, 3+9] = hmd_grot[:, 3]
        else:
            goal_tensor[env_ids, 0+9] = hmd_grot[:, 0]
            goal_tensor[env_ids, 1+9] = hmd_grot[:, 1]
            goal_tensor[env_ids, 2+9] = hmd_grot[:, 2]
            goal_tensor[env_ids, 3+9] = hmd_grot[:, 3]
        #!        

    def reward(self, goal_tensor=None, goal_timer=None):
        if goal_tensor is None: goal_tensor = self.goal_tensor
        if goal_timer is None: goal_timer = self.goal_timer

        p = self.root_pos                                       # 현재 root_pos
        p_ = self.state_hist[-1][:, :3]                         # 이전 root_pos (goal_tensor 구했을 때의 root_pos부터 시작!  / action apply 되기 이전)

        dp_ = goal_tensor[..., :3] - p_                                  # root_pos에서 target 지점까지의 global (dx, dy)
        dp_[:, self.UP_AXIS] = 0
        dist_ = torch.linalg.norm(dp_, ord=2, dim=-1)
        v_ = dp_.div_(goal_timer.unsqueeze(-1)*self.step_time)  # v_: desired veloicty (total distance / sec)

        v_mag = torch.linalg.norm(v_, ord=2, dim=-1)
        sp_ = (dist_/self.step_time).clip_(max=v_mag.clip(min=self.sp_lower_bound, max=self.sp_upper_bound))
        v_ *= (sp_/v_mag).unsqueeze_(-1)                       # desired velocity

        dp = p - p_                                            # (현재 root - 이전 root)
        dp[:, self.UP_AXIS] = 0
        v = dp / self.step_time                                # current velocity: dp / duration 
        r = (v - v_).square_().sum(1).mul_(-3/(sp_*sp_)).exp_()

        dp = goal_tensor[..., :3] - p
        dp[:, self.UP_AXIS] = 0
        dist = torch.linalg.norm(dp, ord=2, dim=-1)
        self.near = dist < self.goal_radius

        r[self.near] = 1
        
        if self.viewer is not None:
            self.goal_timer[self.near] = self.goal_timer[self.near].clip(max=20)
        
        rarm_len, larm_len = self.get_link_len([2,3,4], [3,4,5]), self.get_link_len([2,6,7], [6,7,8])
        rarm_len, larm_len = rarm_len.sum(dim=0), larm_len.sum(dim=0)
        rarm_len, larm_len = rarm_len.repeat(len(self.envs)), larm_len.repeat(len(self.envs))

        #! ego-centric (ee_pos - origin)
        UP_AXIS = 2
        root_pos, root_orient = self.root_pos, self.root_orient                 # [n_envs, 3], [n_envs, 4]
        origin = root_pos.clone()
        origin[..., UP_AXIS] = 0
        heading = heading_zup(root_orient)                                      # N
        up_dir = torch.zeros_like(origin)                                       # N x 1 x 3
        up_dir[..., UP_AXIS] = 1
        heading_orient_inv = axang2quat(up_dir, -heading)                       # N x 4

        #! ADDED rcontrol reward
        target_ego_rhand_pos = goal_tensor[..., :3]
        rhand_pos = self.link_pos[:, self.r_hand_link]
        rcontrol_pos = rhand_pos + rotatepoint(self.link_orient[:, self.l_hand_link], self.l_lpos)
        ego_rhand_pos = rcontrol_pos - origin
        ego_rhand_pos = rotatepoint(heading_orient_inv, ego_rhand_pos)          # N x 3

        e = torch.linalg.norm(target_ego_rhand_pos.sub(ego_rhand_pos), ord=2, dim=-1).div_(rarm_len)
        rhand_rew = e.mul_(-2).exp_()
        #! ADDED rhand reward

        #! ADDED lcontrol reward
        target_ego_lhand_pos = goal_tensor[..., 3:6]
        lhand_pos = self.link_pos[:, self.l_hand_link]
        lcontrol_pos = lhand_pos + rotatepoint(self.link_orient[:, self.l_hand_link], self.l_lpos)
        ego_lhand_pos = lcontrol_pos - origin
        ego_lhand_pos = rotatepoint(heading_orient_inv, ego_lhand_pos)          # N x 3

        l_e = torch.linalg.norm(target_ego_lhand_pos.sub(ego_lhand_pos), ord=2, dim=-1).div_(larm_len)
        lhand_rew = l_e.mul_(-2).exp_()
        #! ADDED lhand reward

        #! ADDED hmd POSITION reward
        target_ego_hmd_pos = goal_tensor[..., 6:9]
        hmd_pos = self.link_pos[:, self.head]
        hmd_pos = hmd_pos + rotatepoint(self.link_orient[:, self.head], self.h_lpos)
        ego_hmd_lpos = hmd_pos - origin
        ego_hmd_lpos = rotatepoint(heading_orient_inv, ego_hmd_lpos)          # N x 3

        hmd_e = torch.linalg.norm(target_ego_hmd_pos.sub(ego_hmd_lpos), ord=2, dim=-1)
        hmd_e_rew = hmd_e.mul_(-3).exp_()
        #! ADDED hmd POSITION reward

        #! ADDED hmd ORIENTATION reward
        target_g_hmd_rot = goal_tensor[..., 9:13]
        head_rot = self.link_orient[:, self.head]
        diff_grot= quat_mul(quat_inverse(head_rot), target_g_hmd_rot)
        ego_diffrot = quat_mul(heading_orient_inv, diff_grot)

        hmd_rot_e = torch.linalg.norm(ego_diffrot, ord=2, dim=-1)
        hmd_rot_e_rew = hmd_rot_e.mul_(-2).exp_()
        #! ADDED hmd ORIENTATION reward

        # print("rhand_rew: ", torch.mean(rhand_rew, dim=0).item(), "lhand_rew: ",  torch.mean(lhand_rew, dim=0).item(), \
        #       "hmd_e_rew: ", torch.mean(hmd_e_rew, dim=0).item(), "hmd_rot_e_rew: ", torch.mean(hmd_rot_e_rew, dim=0).item(),)
        total_r = (self.rpos_coeff * rhand_rew + self.lpos_coeff * lhand_rew \
                    + self.hpos_coeff * hmd_e_rew + self.hrot_coeff * hmd_rot_e_rew)       
        return total_r.unsqueeze_(-1)
    
    def visualize_origin(self):
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, 1.0, 0.0)
        pose.r = gymapi.Quat(0, 0.0, 0.0, 1)
        for i in range(len(self.envs)):
            axes_geom = gymutil.AxesGeometry(1)
            sphere_geom = gymutil.WireframeSphereGeometry(0.04, 16, 16, None, color=(0, 0, 0))   # pink

            gymutil.draw_lines(axes_geom, self.gym, self.viewer, self.envs[i], pose)
            gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], pose)   

        pass

    def visualize_ee_positions(self):
        ee_links = [2, 5, 8, 0]
        hsphere_geom = gymutil.WireframeSphereGeometry(0.04, 16, 16, None, color=(1, 1, 1))     # white
        lsphere_geom = gymutil.WireframeSphereGeometry(0.04, 16, 16, None, color=(1, 0.3, 1))   # pink
        rsphere_geom = gymutil.WireframeSphereGeometry(0.04, 16, 16, None, color=(1, 1, 0.3))   # yellow
        rootsphere_geom = gymutil.WireframeSphereGeometry(0.04, 16, 16, None, color=(0.3, 1, 1))   # pink
        ee_pos = self.goal_link_pos[:, ee_links, :]
        ee_pos = ee_pos        
        for i in range(len(self.envs)):
            head_pos = ee_pos[i, 0]
            rhand_pos = ee_pos[i, 1]
            lhand_pos = ee_pos[i, 2]
            root_pos = ee_pos[i, 3]
            

            head_pose = gymapi.Transform(gymapi.Vec3(head_pos[0], head_pos[1], head_pos[2]), r=None)
            rhand_pose = gymapi.Transform(gymapi.Vec3(rhand_pos[0], rhand_pos[1], rhand_pos[2]), r=None)
            lhand_pose = gymapi.Transform(gymapi.Vec3(lhand_pos[0], lhand_pos[1], lhand_pos[2]), r=None)
            root_pose = gymapi.Transform(gymapi.Vec3(root_pos[0], root_pos[1], root_pos[2]), r=None)


            gymutil.draw_lines(hsphere_geom, self.gym, self.viewer, self.envs[i], head_pose)    # white 
            gymutil.draw_lines(lsphere_geom, self.gym, self.viewer, self.envs[i], lhand_pose)   # pink
            gymutil.draw_lines(rsphere_geom, self.gym, self.viewer, self.envs[i], rhand_pose)
            gymutil.draw_lines(rootsphere_geom, self.gym, self.viewer, self.envs[i], root_pose)   

    def visualize_control_positions(self):
        ee_links = [2, 5, 8, 0]
        ee_pos = self.goal_link_pos[:, ee_links, :]      # [num_envs, 3, ee_links]
        ee_rot = self.goal_link_orient[:, ee_links, :]   # [num_envs, 3, 4]
        lsphere_geom = gymutil.WireframeSphereGeometry(0.04, 16, 16, None, color=(0, 0, 0))   # pink

        curr_ee_rot = self.link_orient[:, ee_links, :]   # [num_envs, 3, 4]
        curr_ee_pos = self.link_pos[:, ee_links, :]

        for i in range(len(self.envs)):
            hmd_pos      = ee_pos[i, 0] + rotatepoint(ee_rot[i, 0], self.h_lpos.to(self.device))
            rcontrol_pos = ee_pos[i, 1] + rotatepoint(ee_rot[i, 1], self.r_lpos.to(self.device))
            lcontrol_pos = ee_pos[i, 2] + rotatepoint(ee_rot[i, 2], self.l_lpos.to(self.device))

            hmd_pose = gymapi.Transform(gymapi.Vec3(hmd_pos[0], hmd_pos[1], hmd_pos[2]), r=None)
            rcontrol_pose = gymapi.Transform(gymapi.Vec3(rcontrol_pos[0], rcontrol_pos[1], rcontrol_pos[2]), r=None)
            lcontrol_pose = gymapi.Transform(gymapi.Vec3(lcontrol_pos[0], lcontrol_pos[1], lcontrol_pos[2]), r=None)
            gymutil.draw_lines(lsphere_geom, self.gym, self.viewer, self.envs[i], hmd_pose)        # black
            gymutil.draw_lines(lsphere_geom, self.gym, self.viewer, self.envs[i], rcontrol_pose)   # black
            gymutil.draw_lines(lsphere_geom, self.gym, self.viewer, self.envs[i], lcontrol_pose)   # black

    def visualize_hmd_rotations(self):
        hmd_position = self.link_pos[:, [2], :] + rotatepoint(self.link_orient[:, [2], :], self.h_lpos.to(self.device))    # [num_envs, frame_num, 4]
        hmd_lrot = self.h_lrot[self.lifetime % 300].unsqueeze(dim=-2)
        hmd_grot = quat_mul(self.link_orient[:, [2], :], hmd_lrot)   # [num_envs, frame_num, 4]
        self.visualize_axis(hmd_position, hmd_grot, scale = 0.2, y=0.2, z =0.2)  # orientation: [num_envs, 1, 4]

    def visualize_goal_positions(self):
        up_ee_links = [3, 4, 5]
        l_key_links = [6, 7, 8]
        lower_key_links = [0, 1, 2, 9, 10, 11, 12, 13, 14]

        lsphere_geom = gymutil.WireframeSphereGeometry(0.04, 16, 16, None, color=(1, 0.3, 1))       # pink
        rsphere_geom = gymutil.WireframeSphereGeometry(0.04, 16, 16, None, color=(1, 1, 0.3))       # yellow
        rootsphere_geom = gymutil.WireframeSphereGeometry(0.04, 16, 16, None, color=(0.3, 1, 1))    # pink

        ee_pos = self.goal_link_pos
        for i in range(len(self.envs)):
            for link in up_ee_links:
                link_pos = ee_pos[i, link]
                link_pose = gymapi.Transform(gymapi.Vec3(link_pos[0], link_pos[1], link_pos[2]), r=None)
                gymutil.draw_lines(rsphere_geom, self.gym, self.viewer, self.envs[i], link_pose)    # white 
            for link in l_key_links:
                link_pos = ee_pos[i, link]
                link_pose = gymapi.Transform(gymapi.Vec3(link_pos[0], link_pos[1], link_pos[2]), r=None)
                gymutil.draw_lines(lsphere_geom, self.gym, self.viewer, self.envs[i], link_pose)    # white
            for link in lower_key_links:
                link_pos = ee_pos[i, link]
                link_pose = gymapi.Transform(gymapi.Vec3(link_pos[0], link_pos[1], link_pos[2]), r=None)
                gymutil.draw_lines(rootsphere_geom, self.gym, self.viewer, self.envs[i], link_pose)    # white

    def visualize_ego_ee(self):
        lsphere_geom = gymutil.WireframeSphereGeometry(0.04, 16, 16, None, color=(1, 0.3, 1))       # pink
        rsphere_geom = gymutil.WireframeSphereGeometry(0.04, 16, 16, None, color=(1, 1, 0.3))       # yellow
        rootsphere_geom = gymutil.WireframeSphereGeometry(0.04, 16, 16, None, color=(0.3, 1, 1))    # pink
        ee_rpos = self.goal_tensor[:, 0:3]
        ee_lpos = self.goal_tensor[:, 3:6]
        root_pos = torch.zeros_like(self.root_pos)
        root_pos[:, 0], root_pos[:, 1] = self.root_pos[:, 0], self.root_pos[:, 1]
        for i in range(len(self.envs)):
            rhand_pos = ee_rpos + root_pos
            lhand_pos = ee_lpos + root_pos
            rhand_pose = gymapi.Transform(gymapi.Vec3(rhand_pos[i, 0], rhand_pos[i, 1], rhand_pos[i, 2]), r=None)
            lhand_pose = gymapi.Transform(gymapi.Vec3(lhand_pos[i, 0], lhand_pos[i, 1], lhand_pos[i, 2]), r=None)
            root_pose = gymapi.Transform(gymapi.Vec3(root_pos[i, 0], root_pos[i, 1], root_pos[i, 2]), r=None)
            gymutil.draw_lines(rsphere_geom, self.gym, self.viewer, self.envs[i], rhand_pose)
            gymutil.draw_lines(lsphere_geom, self.gym, self.viewer, self.envs[i], lhand_pose)
            gymutil.draw_lines(rootsphere_geom, self.gym, self.viewer, self.envs[i], root_pose)

class ICCGANHumanoidVRControl(ICCGANHumanoidVR):
    GOAL_REWARD_WEIGHT = 0.25, 0.25
    GOAL_TENSOR_DIM = (3 + 3 + 3) + (4) + (3)
    GOAL_DIM = 4 + 4 + 4                   # rhand, lhand root's (x, y, sp, dist)   #! should add head?!

    def create_motion_info(self):
        super().create_motion_info()
        #! get user input
        for name, sensorconfig in self.sensor_inputs.items():
            xy_pressed = np.load(os.getcwd() + sensorconfig.xy_pressed)
            xy_pressed = torch.tensor(xy_pressed, dtype=torch.float32, device=self.device)
        #! 
    def _observe(self, env_ids):
        if env_ids is None:
            return observe_iccgan_vrcontrol(
                self.state_hist[-self.ob_horizon:], self.ob_seq_lens,
                self.goal_tensor, self.goal_timer, sp_upper_bound=self.sp_upper_bound, fps=self.fps
            )
        else:
            return observe_iccgan_vrcontrol(
                self.state_hist[-self.ob_horizon:][:, env_ids], self.ob_seq_lens[env_ids],
                self.goal_tensor[env_ids], self.goal_timer[env_ids], sp_upper_bound=self.sp_upper_bound, fps=self.fps
            )
        
    def reset_goal(self, env_ids):
        super().reset_goal(env_ids, self.goal_tensor)
        # self.reset_leg_control_goal(env_ids)

    def overtime_check(self):
        if self.goal_timer is not None:
            self.goal_timer -= 1
            env_ids = torch.nonzero(self.goal_timer <= 0).view(-1)
            # print("self.goal_timer: ", self.goal_timer)
            if len(env_ids) > 0: self.reset_leg_control_goal(env_ids)
        if self.episode_length:
            if callable(self.episode_length):
                return self.lifetime >= self.episode_length(self.simulation_step)
            return self.lifetime >= self.episode_length
        return None

    def reset_leg_control_goal(self, env_ids, goal_timer=None):
        if goal_timer is None: goal_timer = self.goal_timer

        n_envs = len(env_ids)
        all_envs = n_envs == len(self.envs)
        root_orient = self.root_orient if all_envs else self.root_orient[env_ids]

        small_turn = torch.rand(n_envs, device=self.device) > self.sharp_turn_rate                      # 0~1 사이 난수 발생
        large_angle = torch.rand(n_envs, dtype=torch.float32, device=self.device).mul_(2*np.pi)         # 0~2pi 사이
        small_angle = torch.rand(n_envs, dtype=torch.float32, device=self.device).sub_(0.5).mul_(2*(np.pi/3))   

        heading = heading_zup(root_orient)
        small_angle += heading
        theta = torch.where(small_turn, small_angle, large_angle)   # (condition, input, other)

        timer = torch.randint(self.goal_timer_range[0], self.goal_timer_range[1], (n_envs,), dtype=self.goal_timer.dtype, device=self.device)
        # print("timer: ", timer)

        if self.goal_sp_min == self.goal_sp_max:     # juggling+locomotion_walk
            vel = self.goal_sp_min
        elif self.goal_sp_std == 0:                  # juggling+locomotion_walk
            vel = self.goal_sp_mean
        else:
            vel = torch.nn.init.trunc_normal_(torch.empty(n_envs, dtype=torch.float32, device=self.device), mean=self.goal_sp_mean, std=self.goal_sp_std, a=self.goal_sp_min, b=self.goal_sp_max)
        
        dist = vel*timer*self.step_time     # 1/fps에서 얼만큼 갈 수 있는가
        dx = dist*torch.cos(theta)
        dy = dist*torch.sin(theta)

        if all_envs:
            self.init_dist = dist
            goal_timer.copy_(timer)
            self.goal_tensor[:,13] = self.root_pos[:,0] + dx
            self.goal_tensor[:,14] = self.root_pos[:,1] + dy
        else:
            self.init_dist[env_ids] = dist
            goal_timer[env_ids] = timer
            self.goal_tensor[env_ids,13] = self.root_pos[env_ids,0] + dx
            self.goal_tensor[env_ids,14] = self.root_pos[env_ids,1] + dy

    def reward(self, goal_tensor=None, goal_timer=None):
        sensor_tensor = self.goal_tensor[:, :13]
        sensor_rew = super().reward(sensor_tensor)

        control_tensor = self.goal_tensor[:, 13:]

        if goal_timer is None: goal_timer = self.goal_timer

        p = self.root_pos                                       # 현재 root_pos
        p_ = self.state_hist[-1][:, :3]                         # 이전 root_pos (goal_tensor 구했을 때의 root_pos부터 시작!  / action apply 되기 이전)

        dp_ = control_tensor - p_                                  # root_pos에서 target 지점까지의 global (dx, dy)
        dp_[:, self.UP_AXIS] = 0
        dist_ = torch.linalg.norm(dp_, ord=2, dim=-1)
        v_ = dp_.div_(goal_timer.unsqueeze(-1)*self.step_time)  # v_: desired veloicty (total distance / sec)

        v_mag = torch.linalg.norm(v_, ord=2, dim=-1)
        sp_ = (dist_/self.step_time).clip_(max=v_mag.clip(min=self.sp_lower_bound, max=self.sp_upper_bound))
        v_ *= (sp_/v_mag).unsqueeze_(-1)                       # desired velocity

        dp = p - p_                                            # (현재 root - 이전 root)
        dp[:, self.UP_AXIS] = 0
        v = dp / self.step_time                                # current velocity: dp / duration 
        control_rew = (v - v_).square_().sum(1).mul_(-3/(sp_*sp_)).exp_()

        dp = control_tensor - p
        dp[:, self.UP_AXIS] = 0
        dist = torch.linalg.norm(dp, ord=2, dim=-1)
        self.near = dist < self.goal_radius

        control_rew[self.near] = 1
        
        if self.viewer is not None:
            self.goal_timer[self.near] = self.goal_timer[self.near].clip(max=20)

        # control_rew = None
        r = torch.cat((sensor_rew, control_rew.unsqueeze_(-1)), -1)
        return r

    def update_viewer(self):
        super().update_viewer()
        # self.gym.clear_lines(self.viewer)
        n_lines = 10
        tar_x = self.goal_tensor[:, 13].cpu().numpy()

        p = self.root_pos.cpu().numpy()
        zero = np.zeros_like(tar_x)+0.05
        tar_y = self.goal_tensor[:, 14].cpu().numpy()
        lines = np.stack([
            np.stack((p[:,0], p[:,1], zero+0.01*i, tar_x, tar_y, zero), -1)
        for i in range(n_lines)], -2)
        for e, l in zip(self.envs, lines):
            self.gym.add_lines(self.viewer, e, n_lines, l, [[1., 0., 0.] for _ in range(n_lines)])  # red
        n_lines = 10
        target_pos = self.goal_tensor[:, 13:15].cpu().numpy()
        lines = np.stack([
            np.stack((
                target_pos[:, 0], target_pos[:, 1], zero,
                target_pos[:, 0]+self.goal_radius*np.cos(2*np.pi/n_lines*i), 
                target_pos[:, 1]+self.goal_radius*np.sin(2*np.pi/n_lines*i),
                zero
            ), -1)
        for i in range(n_lines)], -2)
        for e, l in zip(self.envs, lines):
            self.gym.add_lines(self.viewer, e, n_lines, l, [[0., 0., 1.] for _ in range(n_lines)])  # blue

@torch.jit.script
def observe_iccgan_vrcontrol(state_hist: torch.Tensor, seq_len: torch.Tensor,
    target_tensor: torch.Tensor, timer: torch.Tensor,
    sp_upper_bound: float, fps: int
):
    sensor_tensor = target_tensor[:, :13]
    target_tensor = target_tensor[:, 13:]

    ob = observe_iccgan_ee(
                state_hist, seq_len,
                sensor_tensor, timer, sp_upper_bound, fps
            )

    #! root position 관련 항목
    root_pos = state_hist[-1, :, :3]
    root_orient = state_hist[-1, :, 3:7]

    dp = target_tensor - root_pos
    x = dp[:, 0]
    y = dp[:, 1]
    heading_inv = -heading_zup(root_orient)
    c = torch.cos(heading_inv)      # root_orientation의 x-dir의 각도 (inverse) 
    s = torch.sin(heading_inv)
    x, y = c*x-s*y, s*x+c*y         # [[c -s], [s c]] * [x y]^T (local_dp -> root_orient에서 바라본 dp)

    dist = (x*x + y*y).sqrt_()
    sp = dist.mul(fps/timer)        # speed! ... dist/timer->how many dist we should go per step ... dist*fps/timer -> how much distance we should go in 1 sec

    too_close = dist < 1e-5
    x = torch.where(too_close, x, x/dist)   # x/dist: normalized x
    y = torch.where(too_close, y, y/dist)
    sp.clip_(max=sp_upper_bound)
    dist.div_(3).clip_(max=1.5)
    #!

    return torch.cat((ob, x.unsqueeze_(-1), y.unsqueeze_(-1), sp.unsqueeze_(-1), dist.unsqueeze_(-1)), -1)

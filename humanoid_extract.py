from typing import Callable, Optional, List, Dict, Any
from collections import namedtuple
import os
from isaacgym import gymapi, gymtorch
import torch
import utils
from utils import heading_zup, axang2quat, rotatepoint, quatconj, quatmultiply, quatdiff_normalized
from isaacgym import gymutil

from ref_motion import ReferenceMotion
from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion
from collections import OrderedDict
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
        self.joint_tensor = joint_tensor.view(len(self.envs), num_dof, -1)  # n_envs x n_dof x 2

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

        ref_motion = {init_pose: self.ref_motion}   # {env_params.motion_file : ref_motion.ReferenceMotion}
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

        # what I added
        self.create_motion_info()
        # what I added
    
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
        # ref_motion.ReferenceMotion, None, 5, DiscriminatorProperty
        # [DiscriminatorProperty(name='walk_in_place/lower', key_links=[0, 9, 10, 11, 12, 13, 14], parent_link=None, local_pos=False, local_height=True, replay_speed=None, ob_horizon=5, id=1)]
        for ref_motion, replay_speed, ob_horizon, discs in self.disc_ref_motion:
            dt = self.step_time
            if replay_speed is not None:
                dt /= replay_speed(n_inst)
            motion_ids, motion_times0 = ref_motion.sample(n_inst, truncate_time=dt*(ob_horizon-1))
            # ob_horizon만큼 motion_ids, motion_times 생성해냄
            motion_ids = np.tile(motion_ids, ob_horizon)    # [motion_ids, motion_ids, ... , motion_ids]
            motion_times = np.concatenate((motion_times0, *[motion_times0+dt*i for i in range(1, ob_horizon)]))
            root_tensor, link_tensor, joint_tensor = ref_motion.state(motion_ids, motion_times)
            real = torch.cat((
                root_tensor, link_tensor.view(root_tensor.size(0), -1)
            ), -1).view(ob_horizon, n_inst, -1)

            for d in discs: samples[d.name] = real
        return self.observe_disc(samples)

    def visualize_axis(self, gpos, gquat):
        gquat = gquat.view(-1, 4).cpu()                                                 # [num_envs x n_links, 4]
        tan_norm = utils.quat_to_tan_norm(gquat).cpu()
        rot_mat = utils.tan_norm_to_rotmat(tan_norm).cpu()
        tan, binorm, norm = rot_mat[..., 0:3], rot_mat[..., 3:6], rot_mat[..., 6:]       # [num_envs x n_links, 3]
        tan, binorm, norm = tan.view(len(self.envs), -1, 3), norm.view(len(self.envs), -1, 3), binorm.view(len(self.envs), -1, 3)   # [num_envs, n_links, 3]
        
        start = gpos.cpu().numpy()                                                      # [5,n_links,3]
        scale = 0.2
        x_end = (gpos.cpu() + tan * scale).cpu().numpy()
        z_end = (gpos.cpu() + binorm * scale).cpu().numpy()
        y_end = (gpos.cpu() + norm * scale).cpu().numpy()
        
        n_lines = 5

        # x-axis
        for j in range(gpos.size(1)):
            x_lines = np.stack([
                np.stack((start[:, j, 0], start[:, j, 1], start[:, j, 2]+0.0015*i, x_end[:, j, 0], x_end[:, j, 1], x_end[:, j, 2]+0.0015*i), -1)
                        for i in range(n_lines)], -2)                                      # [n_envs, n_lines, 6]
            for e, l in zip(self.envs, x_lines):
                self.gym.add_lines(self.viewer, e, n_lines, l, [[1., 0., 0.] for _ in range(n_lines)])

        # y-axis
        for j in range(gpos.size(1)):
            ylines = np.stack([
                np.stack((start[:, j, 0]+0.0015*i, start[:, j, 1], start[:, j, 2], y_end[:, j, 0]+0.0015*i, y_end[:, j, 1], y_end[:, j, 2]), -1)
                        for i in range(n_lines)], -2)                                      # [n_envs, n_lines, 6]
            for e, l in zip(self.envs, ylines):
                self.gym.add_lines(self.viewer, e, n_lines, l, [[0., 1., 0.] for _ in range(n_lines)])

        # z-axis
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

class HumanoidExtract(ICCGANHumanoid):
    
    RANDOM_INIT = False
    def create_motion_info(self):
        self.motion_ids = torch.zeros_like(self.lifetime, dtype=torch.float32, device=self.device)
        self.motion_times = torch.zeros_like(self.lifetime, dtype=torch.float32, device=self.device)


    def update_viewer(self):
        super().update_viewer()
        self.gym.clear_lines(self.viewer)

        self.visualize_ee_positions()


    #! here no forces applied
    def apply_actions(self, actions):
        actions = self.process_actions(actions)
        # make it apply no force
        forces = torch.zeros_like(actions)
        force_tensor = gymtorch.unwrap_tensor(forces)
        self.gym.set_dof_actuation_force_tensor(self.sim, force_tensor)

    def step(self, actions):
        self.state_hist[:-1] = self.state_hist[1:].clone()

        ### Env step
        obs, rews, dones, info = super().step(actions)
        ### Env step END
        info["disc_obs"] = self.observe_disc(self.state_hist)
        info["disc_obs_expert"], info["disc_seq_len"] = self.fetch_real_samples()

        self._motion_sync()
        return obs, rews, dones, info

    def init_state(self, env_ids):
        motion_ids, motion_times = self.ref_motion.sample(len(env_ids))

        self.motion_ids[env_ids] = torch.Tensor(motion_ids)
        self.motion_times[env_ids] = torch.Tensor(motion_times)
        if self.RANDOM_INIT:
            print("random_init time: ", motion_times)
        return self.ref_motion.state(motion_ids, motion_times)
                
    def _motion_sync(self):
        n_inst = len(self.envs)
        env_ids = list(range(n_inst))
        #! 동작이 2개일때 다시 생각해봐야함
        for ref_motion, replay_speed, ob_horizon, discs in self.disc_ref_motion:
            if "upper" in discs[0].name or "full" in discs[0].name:
                # print("ref_motion, replay_speed, ob_horizon, discs: ", ref_motion, replay_speed, ob_horizon, discs)
                dt = self.step_time
                if replay_speed is not None:
                    dt /= replay_speed(n_inst)
                
                # 처음 시작 
                motion_ids, motion_times0 = ref_motion.sample(n_inst, truncate_time=dt*(ob_horizon-1))
                if self.RANDOM_INIT:
                    motion_times0 = self.motion_times + self.lifetime * dt
                else:
                    motion_times0 = self.lifetime * dt
                motion_times0 = motion_times0.cpu().numpy()
                self._ref_root_tensor, self._ref_link_tensor, self._ref_joint_tensor = ref_motion.state(motion_ids, motion_times0)

        n_links = self.gym.get_actor_rigid_body_count(self.envs[0], 0)
        self.root_tensor[env_ids] = self._ref_root_tensor
        self.link_tensor[env_ids] = self._ref_link_tensor
        self.joint_tensor[env_ids] = self._ref_joint_tensor


        print("\n\nself.link_tensor: ", self.link_tensor)

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

    def visualize_ee_positions(self):
        ee_links = [2, 5, 8]
        hsphere_geom = gymutil.WireframeSphereGeometry(0.04, 16, 16, None, color=(1, 1, 1))   # pink
        rsphere_geom = gymutil.WireframeSphereGeometry(0.04, 16, 16, None, color=(1, 1, 0.3))   # yellow
        lsphere_geom = gymutil.WireframeSphereGeometry(0.04, 16, 16, None, color=(1, 0.3, 1))   # pink
        ee_pos = self.link_pos[:, ee_links, :]

        for i in range(len(self.envs)):
            head_pos = ee_pos[i, 0]
            rhand_pos = ee_pos[i, 1]
            lhand_pos = ee_pos[i, 2]

            head_pose = gymapi.Transform(gymapi.Vec3(head_pos[0], head_pos[1], head_pos[2]), r=None)
            rhand_pose = gymapi.Transform(gymapi.Vec3(rhand_pos[0], rhand_pos[1], rhand_pos[2]), r=None)
            lhand_pose = gymapi.Transform(gymapi.Vec3(lhand_pos[0], lhand_pos[1], lhand_pos[2]), r=None)

            gymutil.draw_lines(hsphere_geom, self.gym, self.viewer, self.envs[i], head_pose)   
            gymutil.draw_lines(rsphere_geom, self.gym, self.viewer, self.envs[i], rhand_pose)   
            gymutil.draw_lines(lsphere_geom, self.gym, self.viewer, self.envs[i], lhand_pose)

class ICCGANHumanoidExtractTarget(ICCGANHumanoid):

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

    RANDOM_INIT = True
    MOTION_TIMER_RANGE = 0, 50

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

        self.motion_timer_range = parse_kwarg(kwargs, "motion_timer_range", self.MOTION_TIMER_RANGE)

        super().__init__(*args, **kwargs)

        self.create_motion_info()
        self.create_motion_npy()
    def create_motion_info(self):
        self.motion_ids = torch.zeros(len(self.envs), dtype=torch.int32, device=self.device)
        self.motion_times = torch.zeros(len(self.envs), dtype=torch.float32, device=self.device)

        self.goal_motion_ids = torch.zeros(len(self.envs), dtype=torch.int32, device=self.device)
        self.goal_motion_times = torch.zeros(len(self.envs), dtype=torch.float32, device=self.device)

        self.goal_root_tensor = torch.zeros_like(self.root_tensor, dtype=torch.float32, device=self.device)
        self.goal_link_tensor = torch.zeros_like(self.link_tensor, dtype=torch.float32, device=self.device)
        self.goal_joint_tensor = torch.zeros_like(self.joint_tensor, dtype=torch.float32, device=self.device)
        
        n_links = self.gym.get_actor_rigid_body_count(self.envs[0], 0)
        n_dofs = self.gym.get_actor_dof_count(self.envs[0], 0)
        #reference link tensors and joint tensors
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

    def create_motion_npy(self):
        # get skeleton_tree, is_local, __name__
        self.ref_motion.character_model 
        skeleton = SkeletonTree.from_mjcf(self.character_model).to_dict()
        fps = self.fps
        is_local = False

        self.motion_npy = OrderedDict()
        self.motion_npy['rotation'] = {'arr':None, 'context':{'dtype': 'float64'}}
        self.motion_npy['root_translation'] = {'arr':None, 'context':{'dtype': 'float64'}}
        self.motion_npy['global_velocity'] = {'arr':None, 'context':{'dtype': 'float64'}}
        self.motion_npy['global_angular_velocity'] = {'arr':None, 'context':{'dtype': 'float64'}}
        self.motion_npy['skeleton_tree'] = skeleton
        self.motion_npy['is_local'] = is_local
        self.motion_npy['fps'] = fps
        self.motion_npy['__name__'] = 'SkeletonMotion'
        
        time = self.motion_timer_range[1]-self.motion_timer_range[0]
        
        self.rotation = torch.zeros((time, self.link_tensor.size(1), 4), dtype=torch.float64, device=self.device)                # (58, 15, 4)
        self.root_translation = torch.zeros((time, 3), dtype=torch.float64, device=self.device)                                       # (58, 3)
        self.global_velocity = torch.zeros((time, self.link_tensor.size(1), 3), dtype=torch.float64, device=self.device)         # (58, 15, 3)
        self.global_angular_velocity = torch.zeros((time, self.link_tensor.size(1), 3), dtype=torch.float64, device=self.device) # (58, 15, 3)

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
        
        self.visualize_ee_positions()

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

    def step(self, actions):
        # fill in the otation root_translation, etc for motion_npy
        self._motion_sync()


        if self.lifetime[0] == self.motion_timer_range[1]:
            # save the npy_file
            self.motion_npy['rotation']['arr'] = self.rotation.cpu().numpy() 
            self.motion_npy['root_translation']['arr'] = self.root_translation.cpu().numpy() 
            self.motion_npy['global_velocity']['arr'] = self.global_velocity.cpu().numpy() 
            self.motion_npy['global_angular_velocity']['arr'] = self.global_angular_velocity.cpu().numpy() 
            print("saving self.motion_npy: ", self.motion_npy)
            npy_motion = SkeletonMotion.from_dict(
                dict_repr=self.motion_npy
            )
            # save motion in npy format
            npy_motion.to_file("synth_data/jj_locomotion2.npy")

        obs, rews, dones, info = super().step(actions)

        return obs, rews, dones, info

    def _motion_sync(self):
        n_inst = len(self.envs)
        env_ids = list(range(n_inst))

        self.goal_root_tensor[env_ids] = self.root_tensor
        self.goal_link_tensor[env_ids] = self.link_tensor
        self.goal_joint_tensor[env_ids] = self.joint_tensor

        #motion_timer = torch.randint(self.motion_timer_range[0], self.motion_timer_range[1], (n_inst,), dtype=self.motion_timer.dtype, device=self.device)

        if self.lifetime[0] < self.motion_timer_range[1] and self.lifetime[0] >= self.motion_timer_range[0]:
            # get rotation, root_translation, global_velocity, global_angular_velocity 
            print("self.lifetime: ", self.lifetime[0])
            self.rotation[self.lifetime[0]- self.motion_timer_range[0]] = self.link_tensor[:, :, 3:7] 
            self.root_translation[self.lifetime[0]- self.motion_timer_range[0]] = self.root_tensor[:, 0, :3]
            self.global_velocity[self.lifetime[0]- self.motion_timer_range[0]] = self.root_tensor[:, 0, 7:10]
            self.global_angular_velocity[self.lifetime[0]- self.motion_timer_range[0]] = self.root_tensor[:, 0, 10:13]
        
        ee_links = [5, 8]  # right hand, left hand
        ee_pos = self.goal_link_pos[:, ee_links, :] # [n_envs, n_ee_link, 3]

    def visualize_ee_positions(self):
        ee_links = [2, 5, 8, 0]
        hsphere_geom = gymutil.WireframeSphereGeometry(0.04, 16, 16, None, color=(1, 1, 1))     # white
        lsphere_geom = gymutil.WireframeSphereGeometry(0.04, 16, 16, None, color=(1, 0.3, 1))   # pink
        rsphere_geom = gymutil.WireframeSphereGeometry(0.04, 16, 16, None, color=(1, 1, 0.3))   # yellow
        rootsphere_geom = gymutil.WireframeSphereGeometry(0.04, 16, 16, None, color=(0.3, 1, 1))   # pink
        ee_pos = self.goal_link_pos[:, ee_links, :]
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

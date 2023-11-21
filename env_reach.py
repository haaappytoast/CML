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

class ICCGANHumanoidReach(ICCGANHumanoidVRControl):
    STRIKE_BODY_NAMES = ["right_hand", "right_lower_arm"]

    def __init__(self, *args, 
                sensor_inputs: Optional[Dict[str, SensorInputConfig]]=None,
                 **kwargs):
        self.sensor_inputs = sensor_inputs
        super().__init__(*args, sensor_inputs = self.sensor_inputs, **kwargs)

    def create_envs(self, n: int):
        pass
    
    def _load_obj_asset(self):
        pass
    
    def _build_obj_tensors(self):
        pass

    def _update_obj(self):
        pass     
    
    def update_vobjects(self):
        pass

    def subscribe_keyboards_for_obj(self):
        pass
    
    def update_obj_actions(self, event):
        pass
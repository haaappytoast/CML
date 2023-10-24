import numpy as np
# python main.py config/throw_joystick.py --ckpt 1024_throw_inthewild --headless --server 
env_cls = "ICCGANHumanoidVRControl"
env_params = dict(
    episode_length = 300,
    motion_file = "assets/motions/clips_walk.yaml",    # lower part

    sp_lower_bound = 0.9,
    sp_upper_bound = 1.2,
    goal_timer_range = (90, 150),
    goal_sp_mean = 1.,
    goal_sp_std = 0.25,
    goal_sp_min = 1,
    goal_sp_max = 1.25,
    enableRandomHeading=True,
    goal_termination = False
)

training_params = dict(
    max_epochs = 30000,
    save_interval = 1000,
    terminate_reward = -1
)

reward_coeff = dict(
    rhand_pos = 0.35,       # 3
    lhand_pos = 0.35,       # 3
    hmd_pos = 0.2,          # 3
    hmd_rot = 0.1,          # 4
    heading = 0.7,          # 3
    facing = 0.3            # 2
)

sensor_input = { 
    "train" : dict(
        rlh_localPos = "/assets/retargeted/MetaAvatar@control1@rlh_localPos.npy",
        rlh_localRot = "/assets/retargeted/MetaAvatar@control1@rlh_localRot.npy",
        joystick = "/Unity_postprocess/joystick_input/joystick3"
    ),
    "test" : dict(
        rlh_localPos = "/assets/retargeted/MetaAvatar@control1@rlh_localPos.npy",
        rlh_localRot = "/assets/retargeted/MetaAvatar@control1@rlh_localRot.npy",
        joystick = "/Unity_postprocess/joystick_input/joystick1"
    )
}

discriminators = {
    "usermotion1/upper": dict(
        # motion_file = "assets/retargeted/1020_throw.yaml",
        # motion_file = "assets/retargeted/test/1024_throw/cml@throwjointInfo0 (2).npy",
        motion_file = "assets/retargeted/1024_throw_inthewild.yaml",
        key_links = ["torso", "head", "right_upper_arm", "right_lower_arm", "right_hand", "left_upper_arm", "left_lower_arm", "left_hand"],
        parent_link = "pelvis",
        ob_horizon = 3,
        local_pos = True,
        weight=0.3
    ),
    "walk/lower": dict(
        key_links = ["pelvis", "right_thigh", "right_shin", "right_foot", "left_thigh", "left_shin", "left_foot"],
        parent_link = None,
        weight=0.2
    )
}

# cml@StandingBlockIdle.npy OK --> 너무 잘 학습해버림!
# cml@Blocking1_mirrored.npy OK
# cml@Blocking1.npy NOT OK
# cml@BodyBlock.npy
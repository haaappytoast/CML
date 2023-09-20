import numpy as np
# python main.py config/user_punch+walks.py --ckpt 0918_usermotion/ckpt-9000 --test
env_cls = "ICCGANHumanoidVRControl"
env_params = dict(
    episode_length = 300,
    motion_file = "assets/motions/clips_walk.yaml",    # lower part

    sp_lower_bound = 1,
    sp_upper_bound = 1.2,
    goal_timer_range = (90, 150),
    goal_sp_mean = 1.,
    goal_sp_std = 0.,
    goal_sp_min = 1,
    goal_sp_max = 1
)

training_params = dict(
    max_epochs = 30000,
    save_interval = 300,
    terminate_reward = -1
)

reward_coeff = dict(
    rhand_pos = 1,      # 3
    lhand_pos = 0,      # 3
    hmd_pos = 0,        # 3
    hmd_rot = 0      # 4
)

sensor_input = { 
    "train" : dict(
        rlh_localPos = "/assets/retargeted/MetaAvatar@control1@rlh_localPos.npy",
        rlh_localRot = "/assets/retargeted/MetaAvatar@control1@rlh_localRot.npy",
        xy_pressed = "/assets/retargeted/MetaAvatar@control1@xy_pressed.npy"
    ),
    "test" : dict(
        rlh_localPos = "/assets/retargeted/MetaAvatar@control1@rlh_localPos.npy",
        rlh_localRot = "/assets/retargeted/MetaAvatar@control1@rlh_localRot.npy",
        xy_pressed = "/assets/retargeted/MetaAvatar@control1@xy_pressed.npy"
    )
}

discriminators = {
    "usermotion1/upper": dict(
        motion_file = "assets/retargeted/clips_upperpunch.yaml",
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

import numpy as np
# python main.py config/punch_joystick.py --ckpt 1212_punch_REAL+MIX0302 --headless --server local
env_cls = "ICCGANHumanoidVRControl"
env_params = dict(
    episode_length = 740,
    motion_file = "assets/motions/clips_walk.yaml",    # lower part
    goal_reward_weight = (0.35, 0.15),
    sp_lower_bound = 1.4,
    sp_upper_bound = 1.7,
    goal_timer_range = (90, 150),
    goal_sp_mean = 1.,
    goal_sp_std = 0.25,
    goal_sp_min = 0,
    goal_sp_max = 1.25,
    enableRandomHeading=True,
    goal_termination = False,
    sensor_ablation = False,
    goal_embedding = True,
    eval = False
)

training_params = dict(
    max_epochs = 40000,
    save_interval = 5000,
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
        joystick = "/Unity_postprocess/joystick_input/joystick3"
    )
}

discriminators = {
    "usermotion1/upper": dict(
        # test 5
        #motion_file = "assets/retargeted/1122_punch/1122_punch_MIX+REAL.yaml",
        motion_file = "assets/retargeted/1122_punch/cml@TEST1212punch2_50s_0to740.npy",
        key_links = ["torso", "head", "right_upper_arm", "right_lower_arm", "right_hand", "left_upper_arm", "left_lower_arm", "left_hand"],
        parent_link = "pelvis",
        ob_horizon = 3,
        local_pos = True,
        weight=0.30
    ),
    "walk/lower": dict(
        key_links = ["pelvis", "right_thigh", "right_shin", "right_foot", "left_thigh", "left_shin", "left_foot"],
        parent_link = None,
        weight=0.20
    )
}
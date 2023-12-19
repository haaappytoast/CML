import numpy as np
# python main.py config/pickfruits_reach.py --ckpt 1128_pick_REAL+MIX0302/ckpt-20000 --test
env_cls = "ICCGANHumanoidReach"     # ICCGANHumanoidVRControl
env_params = dict(
    episode_length = 630,               # TEST0: 560 TEST1:630 TEST2: 520
    motion_file = "assets/motions/clips_walk.yaml",    # lower part

    sp_lower_bound = 0.9,
    sp_upper_bound = 1.2,
    goal_timer_range = (90, 150),
    goal_sp_mean = 1.,
    goal_sp_std = 0.25,
    goal_sp_min = 0,
    goal_sp_max = 1.25,
    enableRandomHeading=True,
    #strikeBodyNames = ["right_hand", "right_lower_arm"]
    goal_termination = False,
    goal_embedding = True,
    sensor_ablation = False,
    eval = True
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
        rlh_localPos = "/assets/retargeted/pickfuits_joystick0@rlh_localPos.npy",
        rlh_localRot = "/assets/retargeted/pickfuits_joystick0@rlh_localRot.npy",
        # joystick = "/Unity_postprocess/joystick_input/1016_joystick/joystick_pickfruits1"
        joystick = "/Unity_postprocess/joystick_input/joystick1"
    
    )
}

discriminators = {
    "usermotion1/upper": dict(
        #motion_file = "assets/retargeted/cml@PickFruit_1.npy",
        # motion_file = "assets/retargeted/test/cml@1023_picking_fruits_motion.npy",
        # motion_file = "assets/retargeted/1116_pickup/cml@Picking Up Object_mirrored.npy",
        # motion_file = "assets/retargeted/1116_pickup/1116_pickup_MIX+REAL.yaml",
        motion_file = "assets/retargeted/1116_pickup/cml@user_pick_TEST0.npy",
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

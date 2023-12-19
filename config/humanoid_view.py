# COMMAND: python main.py config/humanoid_view.py --ckpt ckpt_temp --test
env_cls = "HumanoidViewTennis"  #   HumanoidView
env_params = dict(
    episode_length = 450,
    #motion_file = "assets/motions/gym/front_jumping_jack.json"
    # motion_file = "assets/retargeted/1103_tennis/cml@20-Reaction2.npy"
    # motion_file = "assets/retargeted/1116_pickup/cml@user_pick_TEST1.npy",
    # motion_file = "assets/retargeted/1127_tennis/cml@tennis_1127_TRAIN0.npy",
    motion_file = "assets/retargeted/1127_tennis/1106_tennis_locomotion/cml_tennis@run1_subject5_160_2105.npy",
    # motion_file = "assets/retargeted/1114_block/cml@outblock (2).npy",
    #motion_file = "assets/retargeted/1116_pickup/cml@PickFruit_2.npy",    
    #motion_file = "assets/retargeted/1122_punch/cml@user_punch (3).npy",
    # motion_file = "assets/motions/locomotion/lafan1_walk1_subject5_5740_6830.json",

    goal_embedding = False,
    eval = False
)

training_params = dict(
    max_epochs = 10000,
    save_interval = 2000,
    terminate_reward = -1
)

discriminators = {
    "usermotion1/upper": dict(
        parent_link = None,
    )
}
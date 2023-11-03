# COMMAND: python main.py config/humanoid_view.py --ckpt ckpt_temp --test
env_cls = "HumanoidViewTennis"  # HumanoidView
env_params = dict(
    episode_length = 100,
    #motion_file = "assets/motions/gym/front_jumping_jack.json"
    motion_file = "assets/retargeted/1103_tennis/cml@20-Reaction2.npy"
)

training_params = dict(
    max_epochs = 10000,
    save_interval = 2000,
    terminate_reward = -1
)

discriminators = {
    "_/full": dict(
        parent_link = None,
    )
}
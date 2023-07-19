import numpy as np

env_cls = "ICCGANHumanoid"
env_params = dict(
    episode_length = 300,
        motion_file = "assets/motions/gym/front_jumping_jack.json",
)

training_params = dict(
    max_epochs = 30000,
    save_interval = 300,
    terminate_reward = -1
)

discriminators = {
    "front_jumping_jack/lower": dict(
        key_links = ["torso", "head", "right_upper_arm", "right_lower_arm",
                    "pelvis", "right_thigh", "right_shin", "right_foot", "left_thigh", "left_shin", "left_foot"],
        parent_link = None
    ),
    "punch/left_arm": dict(
        motion_file = "assets/motions/iccgan/punch.json",
        key_links = ["left_upper_arm", "left_lower_arm", "left_hand"],
        parent_link = "pelvis",
        local_pos = True,
        replay_speed = lambda n: np.random.uniform(0.8, 1.2, size=(n,))
    )
}

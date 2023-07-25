import numpy as np

env_cls = "ICCGANHumanoidEE_ref"
env_params = dict(
    episode_length = 300,
    motion_file = "assets/motions/gym/chest_open+walk_in_place.json"    # lower part
)

training_params = dict(
    max_epochs = 30000,
    save_interval = 300,
    terminate_reward = -1
)

discriminators = {
    "front_jumping_jack/left_arm": dict(
        motion_file = "assets/motions/gym/front_jumping_jack.json",
        key_links = ["left_upper_arm", "left_lower_arm", "left_hand"],
        parent_link = "pelvis",
        local_pos = True,
        replay_speed = lambda n: np.random.uniform(0.8, 1.2, size=(n,))
    ),
    "punch/upper": dict(
        motion_file = "assets/motions/iccgan/punch.json",
        key_links = ["right_upper_arm", "right_lower_arm", "right_hand"],
        parent_link = "pelvis",
        local_pos = True,
        replay_speed = lambda n: np.random.uniform(0.8, 1.2, size=(n,))
    ),
    "walk_in_place/lower": dict(
        key_links = ["pelvis", "torso", "head", "right_thigh", "right_shin", "right_foot", "left_thigh", "left_shin", "left_foot"],
        parent_link = None
    )
}

# {'head': 2, 'left_foot': 14, 'left_hand': 8, 
# 'left_lower_arm': 7, 'left_shin': 13, 'left_thigh': 12, 
# 'left_upper_arm': 6, 'pelvis': 0, 'right_foot': 11, 
# 'right_hand': 5, 'right_lower_arm': 4, 'right_shin': 10, 
# 'right_thigh': 9, 'right_upper_arm': 3, 'torso': 1} 15
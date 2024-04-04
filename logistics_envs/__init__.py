from gymnasium.envs.registration import register

register(
    id="logistics_envs/VRP-v0",
    entry_point="logistics_envs.envs:VRPEnv",
    max_episode_steps=None,
    autoreset=True,
    kwargs={"max_couriers": 5, "max_nodes": 20},
)

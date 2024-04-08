from gymnasium.envs.registration import register

register(
    id="logistics_envs/QCommerce-v0",
    entry_point="logistics_envs.envs:QCommerceEnv",
    max_episode_steps=None,
    kwargs={
        "start_time": 420,
        "end_time": 1440,
        "time_step": 1,
        "depot_location": [0.5, 0.5],
        "n_couriers": 10,
        "courier_speed": 0.05,
        "max_orders": 40,
        "order_window_size": 20,
        "order_pickup_time": 3,
        "order_drop_off_time": 5,
        "order_generation_start_time": 420,
        "order_generation_end_time": 1320,
        "order_generation_probability": 0.25,
        "max_concurrent_orders": 4,
    },
)

from gymnasium.envs.registration import register

from logistics_envs.sim import LocationMode

register(
    id="logistics_envs/QCommerce-v0",
    entry_point="logistics_envs.envs:QCommerceEnv",
    max_episode_steps=None,
    kwargs={
        "mode": LocationMode.CARTESIAN,
        "start_time": 420,
        "end_time": 1439,
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

register(
    id="logistics_envs/RideHailing-v0",
    entry_point="logistics_envs.envs:RideHailingEnv",
    max_episode_steps=None,
    kwargs={
        "mode": LocationMode.GEOGRAPHIC,
        "start_time": 0,
        "end_time": 1439,
        "time_step": 1,
        "n_drivers": 10,
        "max_orders": 40,
        "order_data_path": "test_data/ride_hailing/ride_hailing_example.csv",
        "order_pickup_time": 1,
        "order_drop_off_time": 1,
        "routing_host": "localhost:8002",
        "render_host": "localhost:8000",
    },
)

__version__ = "0.0.1"

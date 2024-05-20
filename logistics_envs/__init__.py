from gymnasium.envs.registration import register

from logistics_envs.sim import LocationMode
from logistics_envs.envs.ride_hailing_env_config import (
    DriverConfig,
    DriverTravelType,
    RideConfig,
    RideHailingEnvConfig,
)

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
        "config": RideHailingEnvConfig(
            rides=[
                RideConfig(
                    id="ride_1",
                    client_id="client_1",
                    creation_time=0,
                    from_lat=0.5,
                    from_lon=0.5,
                    to_lat=0.6,
                    to_lon=0.6,
                )
            ],
            drivers=[
                DriverConfig(
                    id="driver_1",
                    lat=0.5,
                    lon=0.5,
                    travel_type=DriverTravelType.CAR,
                    speed=0.05,
                    fuel_consumption_rate=0.0,
                ),
            ],
            charging_stations=[],
            start_time=0,
            end_time=1439,
            time_step=1,
            max_rides=10,
            ride_pickup_time=2,
            ride_drop_off_time=1,
            mode=LocationMode.GEOGRAPHIC,
            routing_host="localhost:8002",
            render_mode=None,
            render_host="localhost:8000",
        )
    },
)

__version__ = "0.0.1"

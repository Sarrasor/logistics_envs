import dataclasses
from typing import Optional

import numpy as np
from bidict import bidict
from gymnasium import spaces

from logistics_envs.envs.logistics_sim_wrapper_env import LogisticsSimWrapperEnv
from logistics_envs.envs.ride_hailing_env_config import RideHailingEnvConfig
from logistics_envs.sim.routing_provider import RoutingEngineType
from logistics_envs.sim.structs.action import (
    Action,
    ActionType,
    DeliverActionParameters,
    MoveActionParameters,
    ServiceActionParameters,
    SpecificWorkerAction,
    WorkerAction,
)
from logistics_envs.sim.structs.common import Location, LocationMode
from logistics_envs.sim.structs.config import (
    LogisticsSimulatorConfig,
    OrderConfig,
    RoutingProviderConfig,
    ServiceStationConfig,
)
from logistics_envs.sim.structs.info import Info
from logistics_envs.sim.structs.observation import Observation


class RideHailingEnv(LogisticsSimWrapperEnv):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, config: RideHailingEnvConfig) -> None:
        self._config = config
        self._n_drivers = len(self._config.drivers)
        self._max_rides = self._config.max_rides
        self._n_charging_stations = len(self._config.charging_stations)

        self._worker_id_to_index: bidict[str, int] = bidict()
        self._order_id_to_index: bidict[str, int] = bidict()
        self._station_id_to_index: bidict[str, int] = bidict()

        workers_config = []
        for worker_index, driver_config in enumerate(self._config.drivers):
            self._worker_id_to_index[driver_config.id] = worker_index
            workers_config.append(
                {
                    "id": driver_config.id,
                    "initial_location": {
                        "lat": driver_config.lat,
                        "lon": driver_config.lon,
                    },
                    "travel_type": driver_config.travel_type,
                    "speed": driver_config.speed,
                    "fuel_consumption_rate": driver_config.fuel_consumption_rate,
                }
            )

        orders_config = []
        for ride_config in self._config.rides:
            orders_config.append(
                OrderConfig(
                    id=ride_config.id,
                    client_id=ride_config.client_id,
                    from_location=Location(lat=ride_config.from_lat, lon=ride_config.from_lon),
                    to_location=Location(lat=ride_config.to_lat, lon=ride_config.to_lon),
                    creation_time=ride_config.creation_time,
                    time_window=[(ride_config.creation_time, ride_config.creation_time)],
                ),
            )

        service_stations_config = []
        for station_index, charging_station_config in enumerate(self._config.charging_stations):
            self._station_id_to_index[charging_station_config.id] = station_index
            service_stations_config.append(
                ServiceStationConfig(
                    id=charging_station_config.id,
                    location=Location(
                        lat=charging_station_config.lat, lon=charging_station_config.lon
                    ),
                    service_time=charging_station_config.service_time,
                )
            )

        if self.render_mode == "human":
            if self._config.render_host is None:
                raise ValueError("render_host must be provided when render_mode is human")
            render_config = {
                "render_mode": "WEB",
                "config": {
                    "render_fps": self.metadata["render_fps"],
                    "server_host": self._config.render_host,
                },
            }
        else:
            render_config = {"render_mode": "NONE", "config": None}

        if self._config.mode == LocationMode.GEOGRAPHIC:
            if self._config.routing_host is None:
                raise ValueError("routing_host must be provided when mode is GEOGRAPHIC")
            routing_provider_config = RoutingProviderConfig(
                engine_type=RoutingEngineType.VALHALLA,
                host=self._config.routing_host,
            )
        else:
            routing_provider_config = None

        sim_config = {
            "location_mode": self._config.mode.value,
            "workers": workers_config,
            "order_generator": {
                "generator_type": "PredefinedOrderGenerator",
                "config": {
                    "orders": orders_config,
                },
            },
            "service_stations": service_stations_config,
            "start_time": self._config.start_time,
            "end_time": self._config.end_time,
            "step_size": self._config.time_step,
            "order_pickup_time": self._config.ride_pickup_time,
            "order_drop_off_time": self._config.ride_drop_off_time,
            "render": render_config,
            "routing_provider": routing_provider_config,
            "seed": self._config.seed,
        }
        sim_config = LogisticsSimulatorConfig(**sim_config)
        super().__init__(sim_config)

        if self._config.mode == LocationMode.GEOGRAPHIC:
            # Lat in [-90, 90], lon in [-180, 180]
            location_min = np.array([[-90.0, -180.0]], dtype=np.float32)
            location_max = np.array([[90.0, 180.0]], dtype=np.float32)
        else:
            location_min = np.array([[0.0, 0.0]], dtype=np.float32)
            location_max = np.array([[1.0, 1.0]], dtype=np.float32)

        self.observation_space = spaces.Dict(
            {
                "drivers_location": spaces.Box(
                    low=np.repeat(location_min, self._n_drivers, 0),
                    high=np.repeat(location_max, self._n_drivers, 0),
                    dtype=np.float32,
                ),
                "drivers_status": spaces.MultiDiscrete([6] * self._n_drivers),
                "n_rides": spaces.Discrete(self._max_rides + 1),
                "rides_from_location": spaces.Box(
                    low=np.repeat(location_min, self._max_rides, 0),
                    high=np.repeat(location_max, self._max_rides, 0),
                    dtype=np.float32,
                ),
                "rides_to_location": spaces.Box(
                    low=np.repeat(location_min, self._max_rides, 0),
                    high=np.repeat(location_max, self._max_rides, 0),
                    dtype=np.float32,
                ),
                "rides_status": spaces.MultiDiscrete([6] * self._max_rides),
                "rides_creation_time": spaces.Box(
                    low=0,
                    high=self._config.end_time,
                    shape=(self._max_rides, 1),
                    dtype=np.int32,
                ),
                "charging_stations_location": spaces.Box(
                    low=np.repeat(location_min, self._n_charging_stations, 0),
                    high=np.repeat(location_max, self._n_charging_stations, 0),
                    dtype=np.float32,
                ),
            }
        )
        # action: Primary action. One of: {0=NOOP, 1=MOVE, 2=SERVE, 3=CHARGE}
        # target: Target ride index for SERVE action, target charging station for CHARGE action,
        # and can be arbitrary for NOOP and MOVE
        # Location for MOVE action, can be arbitrary for NOOP, SERVE and CHARGE
        self.action_space = spaces.Dict(
            {
                "action": spaces.MultiDiscrete([4] * self._n_drivers),
                "target": spaces.MultiDiscrete(
                    [max(self._max_rides, self._n_charging_stations)] * self._n_drivers
                ),
                "location": spaces.Box(
                    low=np.repeat(location_min, self._n_drivers, 0),
                    high=np.repeat(location_max, self._n_drivers, 0),
                    dtype=np.float32,
                ),
            }
        )

    def _convert_to_sim_action(self, action_dict: dict) -> Action:
        worker_actions = []
        for worker_index, (action, target, location) in enumerate(
            zip(action_dict["action"], action_dict["target"], action_dict["location"])
        ):
            match action:
                case 0:
                    worker_action = WorkerAction(type=ActionType.NOOP, parameters=None)
                case 1:
                    worker_action = WorkerAction(
                        type=ActionType.MOVE,
                        parameters=MoveActionParameters(
                            to_location=Location(lat=location[0], lon=location[1])
                        ),
                    )
                case 2:
                    worker_action = WorkerAction(
                        type=ActionType.DELIVER,
                        parameters=DeliverActionParameters(
                            order_id=self._order_id_to_index.inverse[target]
                        ),
                    )
                case 3:
                    worker_action = WorkerAction(
                        type=ActionType.SERVICE,
                        parameters=ServiceActionParameters(
                            service_station_id=self._station_id_to_index.inverse[target],
                            # TODO(dburakov): Implement max service time
                            max_service_time=0,
                        ),
                    )
                case _:
                    raise ValueError(f"Unknown action: {action}")

            worker_actions.append(
                SpecificWorkerAction(
                    worker_id=self._worker_id_to_index.inverse[worker_index],
                    action=worker_action,
                )
            )

        return Action(worker_actions=worker_actions)

    def _convert_to_observation(self, sim_observation: Observation) -> dict:
        observation = {
            "drivers_location": np.zeros((self._n_drivers, 2), dtype=np.float32),
            "drivers_status": np.zeros((self._n_drivers,), dtype=np.int32),
            "n_rides": 0,
            "rides_from_location": np.zeros((self._max_rides, 2), dtype=np.float32),
            "rides_to_location": np.zeros((self._max_rides, 2), dtype=np.float32),
            "rides_status": np.zeros((self._max_rides,), dtype=np.int32),
            "rides_creation_time": np.zeros((self._max_rides, 1), dtype=np.int32),
            "charging_stations_location": np.zeros(
                (self._n_charging_stations, 2), dtype=np.float32
            ),
        }
        self._order_id_to_index = bidict()
        n_rides = 0
        for order_observation in sim_observation.orders:
            # TODO(dburakov): Maybe will need to filter order observations that do not
            # have CREATED status

            n_rides += 1
            self._order_id_to_index[order_observation.id] = len(self._order_id_to_index)

            order_index = self._order_id_to_index[order_observation.id]
            observation["rides_from_location"][order_index] = (
                order_observation.from_location.to_numpy()
            )
            observation["rides_to_location"][order_index] = order_observation.to_location.to_numpy()
            observation["rides_status"][order_index] = order_observation.status.to_int()
            observation["rides_creation_time"][order_index] = order_observation.creation_time

            if n_rides >= self._max_rides:
                # Truncate rides if there are more than max_rides
                break
        observation["n_rides"] = n_rides

        for worker_observation in sim_observation.workers:
            worker_index = self._worker_id_to_index[worker_observation.id]
            observation["drivers_location"][worker_index] = worker_observation.location.to_numpy()
            observation["drivers_status"][worker_index] = worker_observation.status.to_int()

        return observation

    def _convert_to_info(self, sim_info: Info) -> dict:
        info = dataclasses.asdict(sim_info)
        return info

    def render(self) -> Optional[dict]:
        return self._sim.render()

    def close(self):
        self._sim.close()

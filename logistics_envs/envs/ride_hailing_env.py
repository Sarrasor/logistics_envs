import dataclasses
import pathlib
from typing import Optional

import numpy as np
import pandas as pd
from bidict import bidict
from gymnasium import spaces

from logistics_envs.envs.logistics_sim_wrapper_env import LogisticsSimWrapperEnv
from logistics_envs.sim.routing_provider import RoutingEngineType
from logistics_envs.sim.structs.action import (
    Action,
    ActionType,
    DeliverActionParameters,
    MoveActionParameters,
    SpecificWorkerAction,
    WorkerAction,
)
from logistics_envs.sim.structs.common import Location, LocationMode
from logistics_envs.sim.structs.config import (
    LogisticsSimulatorConfig,
    OrderConfig,
    RoutingProviderConfig,
)
from logistics_envs.sim.structs.info import Info
from logistics_envs.sim.structs.observation import Observation


class RideHailingEnv(LogisticsSimWrapperEnv):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        mode: LocationMode,
        start_time: int,
        end_time: int,
        time_step: int,
        n_drivers: int,
        max_orders: int,
        order_data_path: str,
        order_pickup_time: int,
        order_drop_off_time: int,
        routing_host: Optional[str] = None,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
        render_host: Optional[str] = None,
    ) -> None:
        if mode != LocationMode.GEOGRAPHIC:
            raise ValueError(
                f"mode={mode} is not supported. Only GEOGRAPHIC mode is supported currently"
            )
        self._mode = mode

        if start_time < 0:
            raise ValueError(f"start_time={start_time} must be greater than or equal to 0")
        self._start_time = start_time
        if end_time <= start_time:
            raise ValueError(f"end_time={end_time} must be greater than start_time={start_time}")
        self._end_time = end_time
        if time_step < 1:
            raise ValueError(f"time_step={time_step} must be greater than 0")
        self._time_step = time_step

        if n_drivers < 1:
            raise ValueError(f"n_drivers={n_drivers} must be greater than 0")
        self._n_drivers = n_drivers

        if max_orders < 1:
            raise ValueError(f"max_orders={max_orders} must be greater than 0")
        self._max_orders = max_orders

        data_path = pathlib.Path(order_data_path)
        if not data_path.exists() or not data_path.suffix == ".csv":
            raise ValueError(f"order_data_path={order_data_path} must be a valid csv file")
        self._order_data_path = order_data_path
        if order_pickup_time < 1:
            raise ValueError(f"order_pickup_time={order_pickup_time} must be greater than 0")
        self._order_pickup_time = order_pickup_time
        if order_drop_off_time < 1:
            raise ValueError(f"order_drop_off_time={order_drop_off_time} must be greater than 0")
        self._order_drop_off_time = order_drop_off_time

        self._routing_host = routing_host

        self._seed = seed
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                f"Invalid render_mode: {render_mode}. Supported modes: {self.metadata['render_modes']}"
            )
        self.render_mode = render_mode
        self._render_host = render_host

        self._worker_id_to_index: bidict[str, int] = bidict()
        self._order_id_to_index: bidict[str, int] = bidict()

        orders_data = pd.read_csv(self._order_data_path)

        workers_config = []
        for worker_index in range(self._n_drivers):
            worker_id = f"driver_{worker_index}"
            self._worker_id_to_index[worker_id] = worker_index
            random_row = orders_data.sample(random_state=self._seed).iloc[0]
            workers_config.append(
                {
                    "id": worker_id,
                    "initial_location": {
                        "lat": random_row["from_lat"],
                        "lon": random_row["from_lon"],
                    },
                    "travel_type": "CAR",
                    "speed": 1.0,
                }
            )

        orders_config = []
        for i, order in orders_data.iterrows():
            orders_config.append(
                OrderConfig(
                    id=f"order_{i}",
                    client_id=f"client_{i}",
                    from_location=Location(lat=order["from_lat"], lon=order["from_lon"]),
                    to_location=Location(lat=order["to_lat"], lon=order["to_lon"]),
                    creation_time=order["creation_time"],
                    time_window=[(order["creation_time"], order["creation_time"])],
                ),
            )

        if self.render_mode == "human":
            if self._render_host is None:
                raise ValueError("render_host must be provided when render_mode is human")
            render_config = {
                "render_mode": "WEB",
                "config": {
                    "render_fps": self.metadata["render_fps"],
                    "server_host": self._render_host,
                },
            }
        else:
            render_config = {"render_mode": "NONE", "config": None}

        if self._mode == LocationMode.GEOGRAPHIC:
            if self._routing_host is None:
                raise ValueError("routing_host must be provided when mode is GEOGRAPHIC")
            routing_provider_config = RoutingProviderConfig(
                engine_type=RoutingEngineType.VALHALLA,
                host=self._routing_host,
            )
        else:
            routing_provider_config = None

        config = {
            "location_mode": self._mode.value,
            "workers": workers_config,
            "order_generator": {
                "generator_type": "PredefinedOrderGenerator",
                "config": {
                    "orders": orders_config,
                },
            },
            "start_time": self._start_time,
            "end_time": self._end_time,
            "step_size": self._time_step,
            "order_pickup_time": self._order_pickup_time,
            "order_drop_off_time": self._order_drop_off_time,
            "render": render_config,
            "routing_provider": routing_provider_config,
            "seed": self._seed,
        }
        config = LogisticsSimulatorConfig(**config)
        super().__init__(config)

        if self._mode == LocationMode.GEOGRAPHIC:
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
                "n_orders": spaces.Discrete(self._max_orders + 1),
                "orders_from_location": spaces.Box(
                    low=np.repeat(location_min, self._max_orders, 0),
                    high=np.repeat(location_max, self._max_orders, 0),
                    dtype=np.float32,
                ),
                "orders_to_location": spaces.Box(
                    low=np.repeat(location_min, self._max_orders, 0),
                    high=np.repeat(location_max, self._max_orders, 0),
                    dtype=np.float32,
                ),
                "orders_status": spaces.MultiDiscrete([6] * self._max_orders),
                "orders_creation_time": spaces.Box(
                    low=0,
                    high=self._end_time,
                    shape=(self._max_orders, 1),
                    dtype=np.int32,
                ),
            }
        )
        # action: Primary action. One of: {0=NOOP, 1=MOVE, 2=SERVE}
        # target: Target order index for MOVE and SERVE actions, can be arbitrary for NOOP
        # Location for MOVE action, can be arbitrary for NOOP and SERVE
        self.action_space = spaces.Dict(
            {
                "action": spaces.MultiDiscrete([3] * self._n_drivers),
                "target": spaces.MultiDiscrete([self._max_orders] * self._n_drivers),
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
            "n_orders": 0,
            "orders_from_location": np.zeros((self._max_orders, 2), dtype=np.float32),
            "orders_to_location": np.zeros((self._max_orders, 2), dtype=np.float32),
            "orders_status": np.zeros((self._max_orders,), dtype=np.int32),
            "orders_creation_time": np.zeros((self._max_orders, 1), dtype=np.int32),
        }
        self._order_id_to_index = bidict()
        n_orders = 0
        for order_observation in sim_observation.orders:
            # TODO(dburakov): Maybe will need to filter order observations that do not
            # have CREATED status

            n_orders += 1
            order_index = len(self._order_id_to_index)
            self._order_id_to_index[order_observation.id] = order_index

            order_index = self._order_id_to_index[order_observation.id]
            observation["orders_from_location"][order_index] = (
                order_observation.from_location.to_numpy()
            )
            observation["orders_to_location"][order_index] = (
                order_observation.to_location.to_numpy()
            )
            observation["orders_status"][order_index] = order_observation.status.to_int()
            observation["orders_creation_time"][order_index] = order_observation.creation_time

            if n_orders >= self._max_orders:
                # Truncate orders if there are more than max_orders
                break
        observation["n_orders"] = n_orders

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

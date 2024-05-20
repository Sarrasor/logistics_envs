import dataclasses
from typing import Optional

import numpy as np
from bidict import bidict
from gymnasium import spaces

from logistics_envs.envs.logistics_sim_wrapper_env import LogisticsSimWrapperEnv
from logistics_envs.sim import (
    Action,
    ActionType,
    DeliverActionParameters,
    Info,
    Location,
    LocationMode,
    LogisticsSimulatorConfig,
    MoveActionParameters,
    Observation,
    SpecificWorkerAction,
    WorkerAction,
)


class QCommerceEnv(LogisticsSimWrapperEnv):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        mode: LocationMode,
        start_time: int,
        end_time: int,
        time_step: int,
        depot_location: tuple[float, float],
        n_couriers: int,
        courier_speed: float,
        max_orders: int,
        order_window_size: int,
        order_pickup_time: int,
        order_drop_off_time: int,
        order_generation_start_time: int,
        order_generation_end_time: int,
        order_generation_probability: float,
        max_concurrent_orders: int,
        routing_host: Optional[str] = None,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
        render_host: Optional[str] = None,
    ):
        if mode != LocationMode.CARTESIAN:
            raise ValueError(
                f"mode={mode} is not supported. Only CARTESIAN mode is supported currently"
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
        self._depot_location = Location(lat=depot_location[0], lon=depot_location[1])

        if n_couriers < 1:
            raise ValueError(f"n_couriers={n_couriers} must be greater than 0")
        self._n_couriers = n_couriers
        if courier_speed <= 0.0:
            raise ValueError(f"courier_speed={courier_speed} must be greater than 0")
        self._courier_speed = courier_speed

        if max_orders < 1:
            raise ValueError(f"max_orders={max_orders} must be greater than 0")
        self._max_orders = max_orders
        if order_window_size < 1:
            raise ValueError(f"order_window_size={order_window_size} must be greater than 0")
        self._order_window_size = order_window_size
        if order_pickup_time < 1:
            raise ValueError(f"order_pickup_time={order_pickup_time} must be greater than 0")
        self._order_pickup_time = order_pickup_time
        if order_drop_off_time < 1:
            raise ValueError(f"order_drop_off_time={order_drop_off_time} must be greater than 0")
        self._order_drop_off_time = order_drop_off_time
        if order_generation_start_time < self._start_time:
            raise ValueError(
                f"order_generation_start_time={order_generation_start_time} must be greater than or equal to start_time={self._start_time}"
            )
        self._order_generation_start_time = order_generation_start_time
        if order_generation_end_time > self._end_time:
            raise ValueError(
                f"order_generation_end_time={order_generation_end_time} must be less than or equal to end_time={self._end_time}"
            )
        self._order_generation_end_time = order_generation_end_time
        if order_generation_probability < 0.0 or order_generation_probability > 1.0:
            raise ValueError(
                f"order_generation_probability={order_generation_probability} must be in the range [0, 1]"
            )
        self._order_generation_probability = order_generation_probability
        if max_concurrent_orders < 1:
            raise ValueError(
                f"max_concurrent_orders={max_concurrent_orders} must be greater than 0"
            )
        self._max_concurrent_orders = max_concurrent_orders

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

        workers_config = []
        for worker_index in range(self._n_couriers):
            worker_id = f"courier_{worker_index}"
            self._worker_id_to_index[worker_id] = worker_index
            workers_config.append(
                {
                    "id": worker_id,
                    "initial_location": self._depot_location.to_dict(),
                    "travel_type": "WALK",
                    "speed": self._courier_speed,
                    "fuel_consumption_rate": 0.0,
                }
            )

        if self.render_mode == "human":
            render_config = {
                "render_mode": "PYGAME",
                "config": {
                    "render_fps": self.metadata["render_fps"],
                    "window_size": (1000, 1000),
                    "hide_completed_orders": True,
                    "bounding_box": {
                        "bottom_left": {"lat": 0.0, "lon": 0.0},
                        "top_right": {"lat": 1.0, "lon": 1.0},
                    },
                },
            }
        else:
            render_config = {"render_mode": "NONE", "config": None}

        config = {
            "location_mode": self._mode.value,
            "workers": workers_config,
            "order_generator": {
                "generator_type": "DepotRandomOrderGenerator",
                "config": {
                    "order_generation_start_time": self._order_generation_start_time,
                    "order_generation_end_time": self._order_generation_end_time,
                    "generation_probability": self._order_generation_probability,
                    "max_concurrent_orders": self._max_concurrent_orders,
                    "depot_location": self._depot_location.to_dict(),
                    "lat_range": (0.0, 1.0),
                    "lon_range": (0.0, 1.0),
                    "window_size": self._order_window_size,
                },
            },
            "service_stations": [],
            "start_time": self._start_time,
            "end_time": self._end_time,
            "step_size": self._time_step,
            "order_pickup_time": self._order_pickup_time,
            "order_drop_off_time": self._order_drop_off_time,
            "render": render_config,
            "seed": self._seed,
        }
        config = LogisticsSimulatorConfig(**config)
        super().__init__(config)

        self.observation_space = spaces.Dict(
            {
                "couriers_location": spaces.Box(
                    low=0.0, high=1.0, shape=(self._n_couriers, 2), dtype=np.float32
                ),
                "couriers_status": spaces.MultiDiscrete([6] * self._n_couriers),
                "depot_location": spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32),
                "n_orders": spaces.Discrete(self._max_orders + 1),
                "orders_from_location": spaces.Box(
                    low=0.0, high=1.0, shape=(self._max_orders, 2), dtype=np.float32
                ),
                "orders_to_location": spaces.Box(
                    low=0.0, high=1.0, shape=(self._max_orders, 2), dtype=np.float32
                ),
                "orders_status": spaces.MultiDiscrete([6] * self._max_orders),
                "orders_creation_time": spaces.Box(
                    low=0,
                    high=self._end_time,
                    shape=(self._max_orders, 1),
                    dtype=np.int32,
                ),
                "orders_time_window": spaces.Box(
                    low=0,
                    high=self._end_time,
                    shape=(self._max_orders, 2),
                    dtype=np.int32,
                ),
            }
        )
        # action: Primary action. One of: {0=NOOP, 1=MOVE, 2=DELIVER}
        # target: Target order index for MOVE and DELIVER actions, can be arbitrary for NOOP
        # Location for MOVE action, can be arbitrary for NOOP and DELIVER
        self.action_space = spaces.Dict(
            {
                "action": spaces.MultiDiscrete([3] * self._n_couriers),
                "target": spaces.MultiDiscrete([self._max_orders] * self._n_couriers),
                "location": spaces.Box(
                    low=0.0, high=1.0, shape=(self._n_couriers, 2), dtype=np.float32
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
            "couriers_location": np.zeros((self._n_couriers, 2), dtype=np.float32),
            "couriers_status": np.zeros((self._n_couriers,), dtype=np.int32),
            "depot_location": self._depot_location.to_numpy(),
            "n_orders": 0,
            "orders_from_location": np.zeros((self._max_orders, 2), dtype=np.float32),
            "orders_to_location": np.zeros((self._max_orders, 2), dtype=np.float32),
            "orders_status": np.zeros((self._max_orders,), dtype=np.int32),
            "orders_creation_time": np.zeros((self._max_orders, 1), dtype=np.int32),
            "orders_time_window": np.zeros((self._max_orders, 2), dtype=np.int32),
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
            observation["orders_time_window"][order_index] = order_observation.time_window[0]

            if n_orders >= self._max_orders:
                # Truncate orders if there are more than max_orders
                break
        observation["n_orders"] = n_orders

        for worker_observation in sim_observation.workers:
            worker_index = self._worker_id_to_index[worker_observation.id]
            observation["couriers_location"][worker_index] = worker_observation.location.to_numpy()
            observation["couriers_status"][worker_index] = worker_observation.status.to_int()

        return observation

    def _convert_to_info(self, sim_info: Info) -> dict:
        info = dataclasses.asdict(sim_info)
        return info

from dataclasses import dataclass
from enum import Enum
from logging import Logger
from typing import TYPE_CHECKING, Optional

import numpy as np

from logistics_envs.sim.structs.action import (
    ActionType,
    DeliverActionParameters,
    DropOffActionParameters,
    MoveActionParameters,
    PickupActionParameters,
    ServiceActionParameters,
    WorkerAction,
)
from logistics_envs.sim.structs.common import Location, LocationMode
from logistics_envs.sim.structs.order import Order

if TYPE_CHECKING:
    from logistics_envs.sim.logistics_simulator import LogisticsSimulator


class WorkerTravelType(str, Enum):
    WALK = "WALK"
    BIKE = "BIKE"
    CAR = "CAR"


class WorkerStatus(str, Enum):
    IDLE = "IDLE"
    MOVING = "MOVING"
    MOVING_TO_PICKUP = "MOVING_TO_PICKUP"
    MOVING_TO_DROP_OFF = "MOVING_TO_DROP_OFF"
    PICKING_UP = "PICKING_UP"
    DROPPING_OFF = "DROPPING_OFF"

    def to_int(self) -> int:
        return self._get_int_from_string(self.value)

    def _get_int_from_string(self, status_str: str) -> int:
        match status_str:
            case "IDLE":
                return 0
            case "MOVING":
                return 1
            case "MOVING_TO_PICKUP":
                return 2
            case "MOVING_TO_DROP_OFF":
                return 3
            case "PICKING_UP":
                return 4
            case "DROPPING_OFF":
                return 5
            case _:
                raise ValueError(f"Unknown status string: {status_str}")

    @staticmethod
    def is_moving_status(status: "WorkerStatus") -> bool:
        return status in {
            WorkerStatus.MOVING,
            WorkerStatus.MOVING_TO_PICKUP,
            WorkerStatus.MOVING_TO_DROP_OFF,
        }


@dataclass(frozen=True)
class WorkerObservation:
    id: str
    location: Location
    travel_type: WorkerTravelType
    speed: float
    status: WorkerStatus
    action: WorkerAction


class Worker:
    def __init__(
        self,
        id: str,
        initial_location: Location,
        travel_type: WorkerTravelType,
        speed: float,
        color: str,
        sim: "LogisticsSimulator",  # noqa: F821 # type: ignore
        logger: Logger,
    ):
        self._id = id
        self._location = initial_location
        self._travel_type = travel_type
        self._speed = speed
        self._color = color
        self._sim = sim
        self._logger = logger

        self._current_action = WorkerAction(type=ActionType.NOOP, parameters=None)
        self._status = WorkerStatus.IDLE
        self._busy_until: Optional[int] = None
        self._path: Optional[dict[int, Location]] = None
        self._picked_up_order_ids: set[str] = set()
        self._current_order_id: Optional[str] = None

    @property
    def id(self) -> str:
        return self._id

    @property
    def is_busy(self) -> bool:
        return self._status != WorkerStatus.IDLE

    @property
    def location(self) -> Location:
        return self._location

    @property
    def color(self) -> str:
        return self._color

    @property
    def status(self) -> WorkerStatus:
        return self._status

    @property
    def busy_until(self) -> Optional[int]:
        return self._busy_until

    @property
    def path(self) -> Optional[dict[int, Location]]:
        return self._path

    def update_state(self, current_time: int) -> None:
        match self._status:
            case WorkerStatus.IDLE:
                self._update_in_idle_status(current_time)
            case (
                WorkerStatus.MOVING
                | WorkerStatus.MOVING_TO_PICKUP
                | WorkerStatus.MOVING_TO_DROP_OFF
            ):
                self._update_in_moving_status(current_time)
            case WorkerStatus.PICKING_UP:
                self._update_in_picking_up_status(current_time)
            case WorkerStatus.DROPPING_OFF:
                self._update_in_dropping_off_status(current_time)
            case _:
                raise ValueError(f"Unknown update worker status {self._status}")

    def _update_in_idle_status(self, current_time: int) -> None:
        self._logger.debug(f"Worker {self._id} is in idle state update")

    def _update_in_moving_status(self, current_time: int) -> None:
        self._logger.debug(f"Worker {self._id} is in moving state update")

        if self._path is None:
            raise ValueError("Worker is in moving state, but path is not set")

        if self._busy_until is None:
            raise ValueError("Worker is in moving state, but busy_until is not set")

        self._location = self._path[current_time]

        if current_time >= self._busy_until:
            match self._status:
                case WorkerStatus.MOVING:
                    self._set_idle_state()
                case WorkerStatus.MOVING_TO_PICKUP:
                    if self._current_order_id is None:
                        raise ValueError(
                            "Worker is in moving to pickup state, but order_id is not set"
                        )
                    self._pickup(self._current_order_id, current_time)
                case WorkerStatus.MOVING_TO_DROP_OFF:
                    if self._current_order_id is None:
                        raise ValueError(
                            "Worker is in moving to drop off state, but order_id is not set"
                        )
                    self._drop_off(self._current_order_id, current_time)
                case _:
                    raise ValueError(f"Unsupported moving status {self._status} in moving update")

    def _update_in_picking_up_status(self, current_time: int) -> None:
        self._logger.debug(f"Worker {self._id} is in picking up state update")

        if self._busy_until is None:
            raise ValueError("Worker is in picking up state, but busy_until is not set")

        if current_time >= self._busy_until:
            if self._current_order_id is None:
                raise ValueError("Worker is in picking up state, but order_id is not set")
            self._picked_up_order_ids.add(self._current_order_id)
            match self._current_action.type:
                case ActionType.PICKUP:
                    self._current_order_id = None
                    self._set_idle_state()
                case ActionType.DELIVER:
                    self._drop_off(self._current_order_id, current_time)
                case _:
                    raise ValueError(
                        f"Unsupported action type {self._current_action.type} in picking up state"
                    )

    def _update_in_dropping_off_status(self, current_time: int) -> None:
        self._logger.debug(f"Worker {self._id} is in dropping off state update")

        if self._busy_until is None:
            raise ValueError("Worker is in drop off state, but busy_until is not set")

        if current_time >= self._busy_until:
            self._set_idle_state()

    def _set_idle_state(self) -> None:
        self._current_action = WorkerAction(type=ActionType.NOOP, parameters=None)
        self._status = WorkerStatus.IDLE
        self._busy_until = None
        self._path = None

    def get_observation(self) -> WorkerObservation:
        return WorkerObservation(
            id=self._id,
            location=self._location,
            travel_type=self._travel_type,
            speed=self._speed,
            status=self._status,
            action=self._current_action,
        )

    def perform_action(self, action: WorkerAction, current_time: int) -> None:
        self._logger.debug(f"Worker {self._id} is performing action {action}")

        # TODO(dburakov): Currently NOOP action does not interrupt the current action. Check if
        # this is the desired behavior
        if action.type == ActionType.NOOP:
            return

        # TODO(dburakov): Check if we should allow changing actions in progress
        if self.is_busy:
            raise ValueError(f"Worker {self._id} is already busy")

        self._current_action = action

        match action.type:
            case ActionType.MOVE:
                if type(action.parameters) is not MoveActionParameters:
                    raise ValueError(f"Invalid move action parameters {action.parameters}")

                self._move(action.parameters.to_location, current_time, WorkerStatus.MOVING)
            case ActionType.DELIVER:
                if type(action.parameters) is not DeliverActionParameters:
                    raise ValueError(f"Invalid deliver action parameters {action.parameters}")

                self._deliver(action.parameters.order_id, current_time)
            case ActionType.PICKUP:
                if type(action.parameters) is not PickupActionParameters:
                    raise ValueError(f"Invalid pickup action parameters {action.parameters}")

                self._pickup(action.parameters.order_id, current_time)
            case ActionType.DROP_OFF:
                if type(action.parameters) is not DropOffActionParameters:
                    raise ValueError(f"Invalid drop off action parameters {action.parameters}")
                self._drop_off(action.parameters.order_id, current_time)
            case ActionType.SERVICE:
                if type(action.parameters) is not ServiceActionParameters:
                    raise ValueError(f"Invalid service action parameters {action.parameters}")

                self._service(
                    action.parameters.service_location,
                    action.parameters.max_service_time,
                    current_time,
                )
            case _:
                raise ValueError(f"Unknown action type {action.type}")

    def _move(self, location: Location, current_time: int, moving_status: WorkerStatus) -> None:
        if self._sim.location_mode == LocationMode.CARTESIAN:
            path, busy_until = self._generate_cartesian_path(location, current_time)
        else:
            path, busy_until = self._generate_geographic_path(location, current_time)

        self._path = path
        self._busy_until = busy_until

        if not WorkerStatus.is_moving_status(moving_status):
            raise ValueError(f"Invalid moving status {moving_status}")

        # TODO(dburakov): Maybe should set status to IDLE if the move is instantaneous
        self._status = moving_status

    def _generate_cartesian_path(
        self, location: Location, current_time: int
    ) -> tuple[dict[int, Location], int]:
        # TODO(dburakov): check if the move instruction is out of bounds

        from_location = self._location.to_numpy()
        to_location = location.to_numpy()
        distance = np.linalg.norm(from_location - to_location)
        dt = self._sim._dt
        travel_time = np.ceil(distance / (dt * self._speed)) * dt
        finish_time = current_time + travel_time
        n_steps = int(travel_time // dt) + 1

        self._logger.debug(
            f"Generating cartesian path from {from_location} to {to_location} Distance: {distance} Current time: {current_time} Travel time: {travel_time} Finish time: {finish_time} N steps: {n_steps}"
        )

        times = np.linspace(current_time, finish_time, n_steps)
        lats = np.linspace(from_location[0], to_location[0], n_steps)
        lons = np.linspace(from_location[1], to_location[1], n_steps)

        path = {}
        for cur_t, cur_lat, cur_lon in zip(times, lats, lons):
            path[cur_t] = Location(lat=cur_lat, lon=cur_lon)

        # For the case, when there is a request to move from a position to the same position
        if distance == 0.0:
            path[times[-1] + dt] = path[times[-1]]

        return path, finish_time

    def _generate_geographic_path(
        self, location: Location, current_time: int
    ) -> tuple[dict[int, Location], int]:
        raise NotImplementedError("Geographic path generation is not implemented")

    def _pickup(self, order_id: str, current_time: int) -> None:
        self._logger.debug(f"Worker {self._id} is picking up order {order_id}")
        order: Order = self._sim.get_order(order_id)
        self._current_order_id = order_id

        if not self._location.near(order.from_location):
            self._logger.debug(f"Worker {self._id} is moving to pickup order {order_id}")
            self._move(order.from_location, current_time, WorkerStatus.MOVING_TO_PICKUP)
        else:
            self._logger.debug(f"Worker {self._id} is calling sim to pick up order {order_id}")
            self._busy_until = self._sim.pickup_order(order.id, self.id)
            self._status = WorkerStatus.PICKING_UP

    def _drop_off(self, order_id: str, current_time: int) -> None:
        self._logger.debug(f"Worker {self._id} is dropping off order {order_id}")
        order: Order = self._sim.get_order(order_id)
        self._current_order_id = order_id

        if not self._location.near(order.to_location):
            self._logger.debug(f"Worker {self._id} is moving to drop off order {order_id}")
            self._move(order.to_location, current_time, WorkerStatus.MOVING_TO_DROP_OFF)
        else:
            self._logger.debug(f"Worker {self._id} is calling sim to drop off order {order_id}")
            self._busy_until = self._sim.drop_off_order(order.id, self.id)
            self._status = WorkerStatus.DROPPING_OFF

    def _deliver(self, order_id: str, current_time: int) -> None:
        order: Order = self._sim.get_order(order_id)

        if order.id in self._picked_up_order_ids:
            self._drop_off(order_id, current_time)
        else:
            self._pickup(order_id, current_time)

    def _service(
        self, service_location: Location, max_service_time: int, current_time: int
    ) -> None:
        raise NotImplementedError("Service action is not implemented")

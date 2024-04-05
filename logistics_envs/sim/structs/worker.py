from dataclasses import dataclass
from enum import Enum
from logging import Logger
from typing import Optional

import numpy as np

# from logistics_envs.sim.logistics_simulator import LogisticsSimulator
from logistics_envs.sim.structs.action import ActionType, MoveActionParameters, WorkerAction
from logistics_envs.sim.structs.common import Location
from logistics_envs.sim.structs.config import LocationMode, WorkerTravelType


class WorkerStatus(str, Enum):
    IDLE = "IDLE"
    MOVING = "MOVING"
    MOVING_TO_PICKUP = "MOVING_TO_PICKUP"
    MOVING_TO_DROP_OFF = "MOVING_TO_DROP_OFF"
    PICKING_UP = "PICKING_UP"
    DROPPING_OFF = "DROPPING_OFF"

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
        sim: "LogisticsSimulator",
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
                    raise NotImplementedError("Moving to pickup transition is not implemented")
                case WorkerStatus.MOVING_TO_DROP_OFF:
                    raise NotImplementedError("Moving to drop off transition is not implemented")
                case _:
                    raise ValueError(f"Unsupported moving status {self._status} in moving update")

    def _update_in_picking_up_status(self, current_time: int) -> None:
        self._logger.debug(f"Worker {self._id} is in picking up state update")

        if self._busy_until is None:
            raise ValueError("Worker is in picking up state, but busy_until is not set")

        if current_time >= self._busy_until:
            match self._current_action.type:
                case ActionType.PICKUP:
                    self._set_idle_state()
                case ActionType.DELIVER:
                    raise NotImplementedError("Deliver transition is not implemented")
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
                self._move(action, current_time, WorkerStatus.MOVING)
            case ActionType.DELIVER:
                self._deliver(action, current_time)
            case ActionType.PICKUP:
                self._pickup(action, current_time)
            case ActionType.DROP_OFF:
                self._drop_off(action, current_time)
            case ActionType.SERVICE:
                self._service(action, current_time)
            case _:
                raise ValueError(f"Unknown action type {action.type}")

    def _move(self, action: WorkerAction, current_time: int, moving_status: WorkerStatus) -> None:
        if type(action.parameters) != MoveActionParameters:
            raise ValueError(
                f"Move action parameters are not of type MoveActionParameters: {action.parameters}"
            )
        parameters: MoveActionParameters = action.parameters
        location = parameters.to_location

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

    def _pickup(self, action: WorkerAction, current_time: int) -> None:
        raise NotImplementedError("Pickup action is not implemented")

    def _drop_off(self, action: WorkerAction, current_time: int) -> None:
        raise NotImplementedError("Drop off action is not implemented")

    def _service(self, action: WorkerAction, current_time: int) -> None:
        raise NotImplementedError("Service action is not implemented")

    def _deliver(self, action: WorkerAction, current_time: int) -> None:
        raise NotImplementedError("Deliver action is not implemented")

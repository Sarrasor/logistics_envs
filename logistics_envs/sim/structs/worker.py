from dataclasses import dataclass
from enum import Enum
from logging import Logger
from typing import Optional
from logistics_envs.sim.structs.action import ActionType, WorkerAction
from logistics_envs.sim.structs.common import Location
from logistics_envs.sim.structs.config import LocationMode, WorkerTravelType


class WorkerStatus(str, Enum):
    IDLE = "IDLE"
    MOVING = "MOVING"
    PICKING_UP = "PICKING_UP"
    DROPPING_OFF = "DROPPING_OFF"


@dataclass(frozen=True)
class WorkerObservation:
    id: str
    location: Location
    travel_type: WorkerTravelType
    status: WorkerStatus
    action: WorkerAction


class Worker:
    def __init__(
        self,
        id: str,
        initial_location: Location,
        travel_type: WorkerTravelType,
        color: str,
        dt: int,
        location_mode: LocationMode,
        logger: Logger,
    ):
        self._id = id
        self._location = initial_location
        self._travel_type = travel_type
        self._color = color
        self._dt = dt
        self._location_mode = location_mode
        self._logger = logger

        self._current_action = WorkerAction(type=ActionType.IDLE, parameters=None)
        self._status = WorkerStatus.IDLE
        self._busy_until: Optional[int] = None
        self._path: Optional[list[Location]] = None

    @property
    def is_busy(self) -> bool:
        return self._current_action.type != ActionType.IDLE

    def update_state(self, current_time: int):
        if self._busy_until and current_time >= self._busy_until:
            self._current_action = WorkerAction(type=ActionType.IDLE, parameters=None)

    def get_observation(self) -> WorkerObservation:
        return WorkerObservation(
            id=self._id,
            location=self._location,
            travel_type=self._travel_type,
            status=self._status,
            action=self._current_action,
        )

    def perform_action(self, action: WorkerAction, current_time: int) -> None:
        self._logger.debug(f"Worker {self._id} is performing action {action}")

        if action.type == ActionType.IDLE:
            self._current_action = action
            self._status = WorkerStatus.IDLE
            self._busy_until = None
            return

        # TODO(dburakov): Allow stopping and changing move actions
        if self.is_busy:
            raise ValueError(f"Worker {self._id} is already busy")

        match action.type:
            case ActionType.MOVE:
                self._move(action, current_time)
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

    def _move(self, action: WorkerAction, current_time: int) -> None:
        raise NotImplementedError("Move action is not implemented")

    def _pickup(self, action: WorkerAction, current_time: int) -> None:
        raise NotImplementedError("Pickup action is not implemented")

    def _drop_off(self, action: WorkerAction, current_time: int) -> None:
        raise NotImplementedError("Drop off action is not implemented")

    def _service(self, action: WorkerAction, current_time: int) -> None:
        raise NotImplementedError("Service action is not implemented")

    def _deliver(self, action: WorkerAction, current_time: int) -> None:
        raise NotImplementedError("Deliver action is not implemented")

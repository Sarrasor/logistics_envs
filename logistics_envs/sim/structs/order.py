from dataclasses import dataclass
from enum import Enum
from typing import Optional

from logistics_envs.sim.structs.common import Location


class OrderStatus(str, Enum):
    CREATED = "CREATED"
    ASSIGNED = "ASSIGNED"
    IN_PICKUP = "IN_PICKUP"
    IN_DELIVERY = "IN_DELIVERY"
    IN_DROP_OFF = "IN_DROP_OFF"
    COMPLETED = "COMPLETED"
    CANCELED = "CANCELED"

    def is_terminal(self) -> bool:
        return self in {OrderStatus.COMPLETED, OrderStatus.CANCELED}

    def to_int(self) -> int:
        return self._get_int_from_string(self.value)

    def _get_int_from_string(self, status_str: str) -> int:
        match status_str:
            case "CREATED":
                return 0
            case "ASSIGNED":
                return 1
            case "IN_PICKUP":
                return 2
            case "IN_DELIVERY":
                return 3
            case "IN_DROP_OFF":
                return 4
            case "COMPLETED":
                return 5
            case "CANCELED":
                return 6
            case _:
                raise ValueError(f"Unknown status string: {status_str}")

    def __eq__(self, other):
        return self.value == other.value

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self._get_int_from_string(self.value) < self._get_int_from_string(other.value)
        return NotImplemented

    def __le__(self, other):
        if self.__class__ is other.__class__:
            return self._get_int_from_string(self.value) <= self._get_int_from_string(other.value)
        return NotImplemented

    def __gt__(self, other):
        if self.__class__ is other.__class__:
            return self._get_int_from_string(self.value) > self._get_int_from_string(other.value)
        return NotImplemented

    def __ge__(self, other):
        if self.__class__ is other.__class__:
            return self._get_int_from_string(self.value) >= self._get_int_from_string(other.value)
        return NotImplemented

    def __hash__(self):
        return hash(self.value)


@dataclass(frozen=True)
class OrderObservation:
    id: str
    client_id: str
    from_location: Location
    to_location: Location
    creation_time: int
    time_window: list[tuple[int, int]]
    status: OrderStatus
    pickup_start_time: Optional[int]
    pickup_end_time: Optional[int]
    drop_off_start_time: Optional[int]
    drop_off_end_time: Optional[int]
    completion_time: Optional[int]


class Order:
    def __init__(
        self,
        id: str,
        client_id: str,
        from_location: Location,
        to_location: Location,
        creation_time: int,
        time_window: list[tuple[int, int]],
    ):
        self._id = id
        self._client_id = client_id
        self._from_location = from_location
        self._to_location = to_location
        self._creation_time = creation_time
        self._time_window = time_window

        self._cancellation_threshold: Optional[int] = None
        self._status = OrderStatus.CREATED
        self._assignment_time: Optional[int] = None
        self._pickup_start_time: Optional[int] = None
        self._pickup_end_time: Optional[int] = None
        self._drop_off_start_time: Optional[int] = None
        self._drop_off_end_time: Optional[int] = None
        self._completion_time: Optional[int] = None
        self._assigned_worker_id = None

    @property
    def id(self) -> str:
        return self._id

    @property
    def client_id(self) -> str:
        return self._client_id

    @property
    def from_location(self) -> Location:
        return self._from_location

    @property
    def to_location(self) -> Location:
        return self._to_location

    @property
    def creation_time(self) -> int:
        return self._creation_time

    @property
    def time_window(self) -> list[tuple[int, int]]:
        return self._time_window

    @property
    def status(self) -> OrderStatus:
        return self._status

    @property
    def pickup_start_time(self) -> Optional[int]:
        return self._pickup_start_time

    @property
    def pickup_end_time(self) -> Optional[int]:
        return self._pickup_end_time

    @property
    def drop_off_start_time(self) -> Optional[int]:
        return self._drop_off_start_time

    @property
    def drop_off_end_time(self) -> Optional[int]:
        return self._drop_off_end_time

    @property
    def completion_time(self) -> Optional[int]:
        return self._completion_time

    @property
    def assignment_time(self) -> Optional[int]:
        return self._assignment_time

    @property
    def assigned_worker_id(self) -> Optional[str]:
        return self._assigned_worker_id

    def set_cancellation_threshold(self, cancellation_threshold: int) -> None:
        self._cancellation_threshold = cancellation_threshold

    def assign(self, worker_id: str, assignment_time: int) -> None:
        self._assigned_worker_id = worker_id
        self._assignment_time = assignment_time
        self._status = OrderStatus.ASSIGNED

    def pickup(self, worker_id: str, pickup_start_time: int, pickup_end_time: int) -> None:
        self._assigned_worker_id = worker_id
        self._pickup_start_time = pickup_start_time
        self._pickup_end_time = pickup_end_time
        self._status = OrderStatus.IN_PICKUP

    def drop_off(self, drop_off_start_time: int, drop_off_end_time: int) -> None:
        self._drop_off_start_time = drop_off_start_time
        self._drop_off_end_time = drop_off_end_time
        self._status = OrderStatus.IN_DROP_OFF

    def update_state(self, current_time: int) -> None:
        # TODO(dburakov): Implement proper cancellation model
        if self._cancellation_threshold is not None and self._status == OrderStatus.CREATED:
            if current_time >= self._creation_time + self._cancellation_threshold:
                self._status = OrderStatus.CANCELED
                self._completion_time = current_time
        elif (
            self._status == OrderStatus.IN_PICKUP
            and self._pickup_end_time is not None
            and current_time >= self._pickup_end_time
        ):
            self._status = OrderStatus.IN_DELIVERY
        elif (
            self._status == OrderStatus.IN_DROP_OFF
            and self._drop_off_end_time is not None
            and current_time >= self._drop_off_end_time
        ):
            # TODO(dburakov): Actual time transition can happen some time after real drop off time
            self._status = OrderStatus.COMPLETED
            self._completion_time = self._drop_off_end_time

    def get_observation(self, include_intermediate: bool = False) -> OrderObservation:
        pickup_start_time = None
        pickup_end_time = None
        drop_off_start_time = None
        drop_off_end_time = None

        if include_intermediate:
            if self._status != OrderStatus.CREATED:
                pickup_start_time = self._pickup_start_time

            if self._status > OrderStatus.IN_PICKUP:
                pickup_end_time = self._pickup_end_time

            if self._status > OrderStatus.IN_DELIVERY:
                drop_off_start_time = self._drop_off_start_time

            if self._status > OrderStatus.IN_DROP_OFF:
                drop_off_end_time = self._drop_off_end_time

        return OrderObservation(
            id=self._id,
            client_id=self._client_id,
            from_location=self._from_location,
            to_location=self._to_location,
            creation_time=self._creation_time,
            time_window=self._time_window,
            status=self._status,
            pickup_start_time=pickup_start_time,
            pickup_end_time=pickup_end_time,
            drop_off_start_time=drop_off_start_time,
            drop_off_end_time=drop_off_end_time,
            completion_time=self._completion_time,
        )

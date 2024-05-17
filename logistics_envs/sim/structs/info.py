from dataclasses import dataclass
from typing import Optional

from logistics_envs.sim.structs.common import Location
from logistics_envs.sim.structs.order import OrderStatus
from logistics_envs.sim.structs.worker import WorkerTravelType


@dataclass
class OrderInfo:
    id: str
    client_id: str
    from_location: Location
    to_location: Location
    creation_time: int
    time_window: list[tuple[int, int]]
    status: OrderStatus
    assignment_time: Optional[int]
    pickup_start_time: Optional[int]
    pickup_end_time: Optional[int]
    drop_off_start_time: Optional[int]
    drop_off_end_time: Optional[int]
    completion_time: Optional[int]
    assigned_worker_id: Optional[str]


@dataclass
class WorkerInfo:
    id: str
    travel_type: WorkerTravelType
    speed: float
    color: str


@dataclass
class Metric:
    name: str
    value: float
    unit: str


@dataclass(frozen=True)
class Info:
    start_time: int
    end_time: int
    orders: list[OrderInfo]
    workers: list[WorkerInfo]
    metrics: list[Metric]

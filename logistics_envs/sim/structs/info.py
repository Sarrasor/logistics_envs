from dataclasses import dataclass
from typing import Optional

from logistics_envs.sim.structs.common import Location
from logistics_envs.sim.structs.order import OrderStatus
from logistics_envs.sim.structs.worker import WorkerStatus, WorkerTravelType


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
    status_history: list[tuple[int, WorkerStatus]]
    n_assigned_orders: int
    n_completed_orders: int
    completion_rate: float
    idle_rate: float
    with_order_rate: float
    traveled_distance: float
    consumed_fuel: float
    n_service_station_visits: int


@dataclass
class ServiceStationInfo:
    id: str
    location: Location
    service_events: list[tuple[str, int, int]]


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
    service_stations: list[ServiceStationInfo]
    metrics: list[Metric]

from dataclasses import dataclass
from logistics_envs.sim.structs.order import OrderObservation
from logistics_envs.sim.structs.service_station import ServiceStationObservation
from logistics_envs.sim.structs.worker import WorkerObservation


@dataclass(frozen=True)
class Observation:
    current_time: int
    workers: list[WorkerObservation]
    orders: list[OrderObservation]
    service_stations: list[ServiceStationObservation]

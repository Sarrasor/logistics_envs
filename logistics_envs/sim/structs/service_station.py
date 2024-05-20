from dataclasses import dataclass
from logistics_envs.sim.structs.common import Location


@dataclass(frozen=True)
class ServiceStationObservation:
    id: str
    location: Location


@dataclass(frozen=True)
class ServiceEvent:
    worker_id: str
    max_service_time: int
    start_time: int
    end_time: int


class ServiceStation:
    def __init__(self, id: str, location: Location, service_time: int) -> None:
        self._id = id
        self._location = location
        self._service_time = service_time

        self._service_events = []

    @property
    def id(self) -> str:
        return self._id

    @property
    def location(self) -> Location:
        return self._location

    @property
    def service_time(self) -> int:
        return self._service_time

    @property
    def service_events(self) -> list[ServiceEvent]:
        return self._service_events.copy()

    def get_observation(self) -> ServiceStationObservation:
        return ServiceStationObservation(
            id=self.id,
            location=self.location,
        )

    def service(self, worker_id: str, max_service_time: int, current_time: int) -> int:
        # TODO(dburakov): Add service time sampler
        # TODO(dburakov): Utilize max_service_time
        service_end_time = current_time + self.service_time
        self._service_events.append(
            ServiceEvent(worker_id, max_service_time, current_time, service_end_time)
        )

        return service_end_time

from enum import Enum
from typing import Optional

import polyline
import requests

from logistics_envs.sim.structs.common import Location, Route
from logistics_envs.sim.structs.worker import WorkerTravelType


class RoutingEngineType(str, Enum):
    VALHALLA = "VALHALLA"
    OSRM = "OSRM"


class RoutingProvider:
    def __init__(self, engine_type: RoutingEngineType, host: str) -> None:
        self._engine_type = engine_type
        self._host = host

    def get_route(
        self,
        from_location: Location,
        to_location: Location,
        worker_travel_type: Optional[WorkerTravelType] = None,
    ) -> Route:
        if self._engine_type == RoutingEngineType.VALHALLA:
            return self._get_route_with_valhalla(from_location, to_location, worker_travel_type)
        elif self._engine_type == RoutingEngineType.OSRM:
            return self._get_route_with_osrm(from_location, to_location, worker_travel_type)
        else:
            raise NotImplementedError(f"Unsupported routing engine type: {self._engine_type}")

    def _get_route_with_valhalla(
        self,
        from_location: Location,
        to_location: Location,
        worker_travel_type: Optional[WorkerTravelType] = None,
    ) -> Route:
        costing = "auto"
        if worker_travel_type is not None:
            match worker_travel_type:
                case WorkerTravelType.CAR:
                    costing = "auto"
                case WorkerTravelType.BIKE:
                    costing = "bicycle"
                case WorkerTravelType.WALK:
                    costing = "pedestrian"

        payload = {
            "locations": [from_location.to_dict(), to_location.to_dict()],
            "costing": costing,
            "units": "km",
        }
        response = requests.post(f"http://{self._host}/route", json=payload, timeout=1.0)
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to get route with Valhalla. Response status code: {response.status_code}"
            )

        response = response.json()
        response["trip"]["legs"][0]["shape"]

        geometry = response["trip"]["legs"][0]["shape"]
        # google-polyline library can only decode precision 5
        geometry = polyline.encode(polyline.decode(geometry, precision=6), precision=5)

        route = Route(
            geometry=geometry,
            precision=5,
            length_meters=response["trip"]["summary"]["length"] * 1000.0,
            time_seconds=response["trip"]["summary"]["time"],
        )

        return route

    def _get_route_with_osrm(
        self,
        from_location: Location,
        to_location: Location,
        worker_travel_type: Optional[WorkerTravelType] = None,
    ) -> Route:
        if worker_travel_type is not None and worker_travel_type != WorkerTravelType.CAR:
            raise NotImplementedError(
                f"Currently only CAR travel type is supported for OSRM. Received {worker_travel_type}"
            )

        coords = [
            f"{from_location.lon},{from_location.lat}",
            f"{to_location.lon},{to_location.lat}",
        ]
        payload = ";".join(coords)
        response = requests.get(
            f"http://{self._host}/route/v1/car/{payload}?overview=full&continue_straight=false",
            timeout=1.0,
        )
        response = response.json()

        if response["code"] != "Ok":
            raise RuntimeError(f"Failed to get route with OSRM. Response code: {response['code']}")

        geometry = response["routes"][0]["geometry"]
        route = Route(
            geometry=geometry,
            precision=5,
            length_meters=response["routes"][0]["distance"],
            time_seconds=response["routes"][0]["duration"],
        )

        return route

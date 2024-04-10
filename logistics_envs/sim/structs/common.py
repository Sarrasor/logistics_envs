from dataclasses import dataclass
from enum import Enum

import numpy as np
from geopy.distance import geodesic
import polyline


@dataclass
class Location:
    lat: float
    lon: float

    def to_dict(self) -> dict:
        return {"lat": self.lat, "lon": self.lon}

    def to_numpy(self) -> np.ndarray:
        return np.array([self.lat, self.lon], dtype=np.float32)

    def __hash__(self):
        return hash((self.lat, self.lon))

    def near(self, other: "Location", threshold: float = 1e-4) -> bool:
        return abs(self.lat - other.lat) < threshold and abs(self.lon - other.lon) < threshold

    def distance_to(self, other: "Location") -> float:
        """
        Calculates the distance between two locations using the geodesic formula.

        Parameters:
            other (Location): The other location to calculate the distance to.

        Returns:
            float: The distance between the two locations in meters.
        """
        return geodesic((self.lat, self.lon), (other.lat, other.lon)).meters


@dataclass
class Route:
    geometry: str
    precision: int
    length_meters: float
    time_seconds: float

    def get_points(self) -> list[Location]:
        return [Location(*p) for p in polyline.decode(self.geometry, precision=self.precision)]


class LocationMode(str, Enum):
    CARTESIAN = "CARTESIAN"  # Use locations as cartesian coordinates
    GEOGRAPHIC = "GEOGRAPHIC"  # Use locations as geographic coordinates


@dataclass
class BoundingBox:
    bottom_left: Location
    top_right: Location

from dataclasses import dataclass

import numpy as np


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


@dataclass
class BoundingBox:
    bottom_left: Location
    top_right: Location

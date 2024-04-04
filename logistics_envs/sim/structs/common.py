from dataclasses import dataclass


@dataclass
class Location:
    lat: float
    lon: float

    def get_dict(self) -> dict:
        return {"lat": self.lat, "lon": self.lon}

    def __hash__(self):
        return hash((self.lat, self.lon))

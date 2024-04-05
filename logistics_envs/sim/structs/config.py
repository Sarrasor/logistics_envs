from dataclasses import dataclass
from enum import Enum
from typing import Optional

from logistics_envs.sim.structs.common import BoundingBox, Location


class WorkerTravelType(str, Enum):
    WALK = "WALK"
    BIKE = "BIKE"
    CAR = "CAR"


@dataclass
class WorkerConfig:
    id: str
    initial_location: Location
    travel_type: WorkerTravelType
    speed: float


class LocationMode(str, Enum):
    CARTESIAN = "CARTESIAN"  # Use locations as cartesian coordinates
    GEOGRAPHIC = "GEOGRAPHIC"  # Use locations as geographic coordinates


@dataclass
class OrderConfig:
    id: str
    client_id: str
    from_location: Location
    to_location: Location
    creation_time: int
    time_window: list[tuple[int, int]]


@dataclass
class PredefinedOrderGeneratorConfig:
    orders: list[OrderConfig]


@dataclass
class RandomOrderGeneratorConfig:
    generation_probability: float
    max_concurrent_orders: int
    allow_same_location: bool
    lat_range: tuple[float, float]
    lon_range: tuple[float, float]
    window_size: int


@dataclass
class OrderGeneratorConfig:
    generator_type: str
    config: PredefinedOrderGeneratorConfig | RandomOrderGeneratorConfig


class RenderMode(str, Enum):
    NONE = "NONE"
    JSON = "JSON"
    PYGAME = "PYGAME"
    WEB = "WEB"


@dataclass
class PygameRenderConfig:
    render_fps: int
    window_size: tuple[int, int]
    hide_completed_orders: bool
    bounding_box: BoundingBox


@dataclass
class RenderConfig:
    render_mode: RenderMode
    config: Optional[PygameRenderConfig] = None


@dataclass
class LogisticsSimulatorConfig:
    location_mode: LocationMode
    workers: list[WorkerConfig]
    order_generator: OrderGeneratorConfig
    start_time: int
    end_time: int
    step_size: int
    # TODO(dburakov): add pickup and drop_off samplers
    order_pickup_time: int
    order_drop_off_time: int
    render: RenderConfig
    seed: Optional[int] = None

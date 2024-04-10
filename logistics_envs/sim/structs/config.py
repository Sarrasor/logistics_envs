from enum import Enum
from typing import Optional

from pydantic import (
    BaseModel,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
)

from logistics_envs.sim.structs.common import BoundingBox, Location, LocationMode
from logistics_envs.sim.structs.worker import WorkerTravelType
from logistics_envs.sim.routing_provider import RoutingEngineType


class WorkerConfig(BaseModel):
    id: str
    initial_location: Location
    travel_type: WorkerTravelType
    speed: PositiveFloat


class OrderConfig(BaseModel):
    id: str
    client_id: str
    from_location: Location
    to_location: Location
    creation_time: NonNegativeInt
    time_window: list[tuple[NonNegativeInt, NonNegativeInt]]


class PredefinedOrderGeneratorConfig(BaseModel):
    orders: list[OrderConfig]


class RandomOrderGeneratorConfig(BaseModel):
    generation_probability: NonNegativeFloat
    max_concurrent_orders: PositiveInt
    allow_same_location: bool
    lat_range: tuple[float, float]
    lon_range: tuple[float, float]
    window_size: PositiveInt


class DepotRandomOrderGeneratorConfig(BaseModel):
    order_generation_start_time: NonNegativeInt
    order_generation_end_time: NonNegativeInt
    generation_probability: NonNegativeFloat
    max_concurrent_orders: PositiveInt
    depot_location: Location
    lat_range: tuple[float, float]
    lon_range: tuple[float, float]
    window_size: PositiveInt


class OrderGeneratorConfig(BaseModel):
    generator_type: str
    config: (
        PredefinedOrderGeneratorConfig
        | RandomOrderGeneratorConfig
        | DepotRandomOrderGeneratorConfig
    )


class RenderMode(str, Enum):
    NONE = "NONE"
    JSON = "JSON"
    PYGAME = "PYGAME"
    WEB = "WEB"


class PygameRenderConfig(BaseModel):
    render_fps: PositiveInt
    window_size: tuple[PositiveInt, PositiveInt]
    hide_completed_orders: bool
    bounding_box: BoundingBox


class WebRenderConfig(BaseModel):
    render_fps: PositiveInt
    server_host: str


class RenderConfig(BaseModel):
    render_mode: RenderMode
    config: Optional[PygameRenderConfig | WebRenderConfig] = None


class RoutingProviderConfig(BaseModel):
    engine_type: RoutingEngineType
    host: str


class LogisticsSimulatorConfig(BaseModel):
    location_mode: LocationMode
    workers: list[WorkerConfig]
    order_generator: OrderGeneratorConfig
    start_time: NonNegativeInt
    end_time: NonNegativeInt
    step_size: PositiveInt
    # TODO(dburakov): add pickup and drop_off samplers
    order_pickup_time: NonNegativeInt
    order_drop_off_time: NonNegativeInt
    render: RenderConfig
    routing_provider: Optional[RoutingProviderConfig] = None
    seed: Optional[int] = None

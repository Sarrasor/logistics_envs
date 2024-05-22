from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, NonNegativeInt, PositiveInt, field_validator

from logistics_envs.sim.structs.common import LocationMode


class RideConfig(BaseModel):
    id: str
    client_id: str
    creation_time: NonNegativeInt
    from_lat: float
    from_lon: float
    to_lat: float
    to_lon: float


class DriverTravelType(str, Enum):
    WALK = "WALK"
    BIKE = "BIKE"
    CAR = "CAR"


class DriverConfig(BaseModel):
    id: str
    lat: float
    lon: float
    travel_type: DriverTravelType
    speed: float
    fuel_consumption_rate: float


class ChargingStationConfig(BaseModel):
    id: str
    lat: float
    lon: float
    service_time: NonNegativeInt


class RideHailingEnvConfig(BaseModel):
    rides: list[RideConfig]
    drivers: list[DriverConfig]
    charging_stations: list[ChargingStationConfig]
    start_time: NonNegativeInt
    end_time: NonNegativeInt
    time_step: PositiveInt
    max_rides: PositiveInt
    ride_pickup_time: NonNegativeInt
    ride_drop_off_time: NonNegativeInt
    mode: LocationMode
    routing_host: Optional[str] = None
    seed: Optional[int] = None
    order_cancellation_threshold: Optional[NonNegativeInt] = None
    render_mode: Optional[str] = None
    render_host: Optional[str] = None

    @field_validator("rides")
    @classmethod
    def at_least_one_ride(cls, v):
        if len(v) == 0:
            raise ValueError("At least one ride must be provided")
        return v

    @field_validator("drivers")
    @classmethod
    def at_least_one_driver(cls, v):
        if len(v) == 0:
            raise ValueError("At least one driver must be provided")
        return v

    @staticmethod
    def from_file(
        file_path: str,
        mode: LocationMode,
        routing_host: Optional[str],
        render_mode: Optional[str],
        render_host: Optional[str],
    ) -> "RideHailingEnvConfig":
        rides_data = pd.read_excel(file_path, sheet_name="rides")
        rides = []
        for _, row in rides_data.iterrows():
            rides.append(
                RideConfig(
                    id=row["id"],
                    client_id=row["client_id"],
                    creation_time=row["creation_time"],
                    from_lat=row["from_lat"],
                    from_lon=row["from_lon"],
                    to_lat=row["to_lat"],
                    to_lon=row["to_lon"],
                )
            )

        drivers_data = pd.read_excel(file_path, sheet_name="drivers")
        drivers = []
        for _, row in drivers_data.iterrows():
            drivers.append(
                DriverConfig(
                    id=row["id"],
                    lat=row["lat"],
                    lon=row["lon"],
                    travel_type=DriverTravelType(row["travel_type"]),
                    speed=row["speed"],
                    fuel_consumption_rate=row["fuel_consumption_rate"],
                )
            )

        charging_stations_data = pd.read_excel(file_path, sheet_name="charging_stations")
        charging_stations = []
        for _, row in charging_stations_data.iterrows():
            charging_stations.append(
                ChargingStationConfig(
                    id=row["id"],
                    lat=row["lat"],
                    lon=row["lon"],
                    service_time=row["service_time"],
                )
            )

        config_data = pd.read_excel(file_path, sheet_name="config")
        config_data = config_data.replace({np.nan: None})
        config_data = config_data.iloc[0]

        return RideHailingEnvConfig(
            rides=rides,
            drivers=drivers,
            charging_stations=charging_stations,
            start_time=config_data["start_time"],
            end_time=config_data["end_time"],
            time_step=config_data["time_step"],
            max_rides=config_data["max_rides"],
            ride_pickup_time=config_data["ride_pickup_time"],
            ride_drop_off_time=config_data["ride_drop_off_time"],
            seed=config_data["seed"],
            order_cancellation_threshold=config_data["order_cancellation_threshold"],
            mode=mode,
            routing_host=routing_host,
            render_mode=render_mode,
            render_host=render_host,
        )

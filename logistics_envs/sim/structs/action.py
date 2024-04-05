from dataclasses import dataclass
from enum import Enum
from logistics_envs.sim.structs.common import Location


class ActionType(str, Enum):
    NOOP = "NOOP"  # Do nothing
    MOVE = "MOVE"  # Move to a location
    DELIVER = "DELIVER"  # Deliver an order. Performs move, pickup and drop off actions by itself
    PICKUP = "PICKUP"  # Pick up an order from a location. Can be used for multi-order delivery
    DROP_OFF = "DROP_OFF"  # Drop off an order at a location. Can be used for multi-order delivery
    SERVICE = "SERVICE"  # Perform service at a location. For example, clean the vehicle or refuel


@dataclass
class MoveActionParameters:
    to_location: Location


@dataclass
class DeliverActionParameters:
    order_id: str


@dataclass
class PickupActionParameters:
    order_id: str


@dataclass
class DropOffActionParameters:
    order_id: str


@dataclass
class ServiceActionParameters:
    # TODO(dburakov): maybe should replace location with service_id
    service_location: Location
    max_service_time: int


@dataclass
class WorkerAction:
    type: ActionType
    parameters: (
        None
        | MoveActionParameters
        | DeliverActionParameters
        | PickupActionParameters
        | DropOffActionParameters
        | ServiceActionParameters
    )


@dataclass
class SpecificWorkerAction:
    worker_id: str
    action: WorkerAction


@dataclass
class Action:
    worker_actions: list[SpecificWorkerAction]

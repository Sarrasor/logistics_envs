import logging
import uuid
from typing import Optional

import polyline
import pygame
import requests

from logistics_envs.sim import order_generators
from logistics_envs.sim.routing_provider import RoutingProvider
from logistics_envs.sim.structs.action import Action
from logistics_envs.sim.structs.common import Location
from logistics_envs.sim.structs.config import (
    LocationMode,
    LogisticsSimulatorConfig,
    PygameRenderConfig,
    RenderMode,
    WebRenderConfig,
)
from logistics_envs.sim.structs.info import Info, OrderInfo, ServiceStationInfo, WorkerInfo
from logistics_envs.sim.structs.observation import Observation
from logistics_envs.sim.structs.order import Order, OrderStatus
from logistics_envs.sim.structs.service_station import ServiceStation
from logistics_envs.sim.structs.worker import Worker, WorkerStatus
from logistics_envs.sim.utils import generate_colors, set_global_seeds

logger = logging.getLogger("logistics_simulator")


class LogisticsSimulator:
    def __init__(self, config: LogisticsSimulatorConfig) -> None:
        self._config = config
        self._location_mode = self._config.location_mode
        if self._location_mode == LocationMode.GEOGRAPHIC:
            if self._config.routing_provider is None:
                raise ValueError("Geographic location mode requires routing provider config")

            self._routing_provider = RoutingProvider(
                engine_type=self._config.routing_provider.engine_type,
                host=self._config.routing_provider.host,
            )

        self._dt = self._config.step_size

        self._render_mode = self._config.render.render_mode
        if self._render_mode == RenderMode.PYGAME:
            if not isinstance(self._config.render.config, PygameRenderConfig):
                raise ValueError(
                    f"Pygame render mode requires PygameRenderConfig, but got {self._config.render.config}"
                )

            render_config: PygameRenderConfig = self._config.render.config
            self._render_fps = render_config.render_fps
            self._window_size = render_config.window_size
            self._x_scale = self._window_size[0] / (
                render_config.bounding_box.top_right.lon
                - render_config.bounding_box.bottom_left.lon
            )
            self._y_scale = self._window_size[1] / (
                render_config.bounding_box.top_right.lat
                - render_config.bounding_box.bottom_left.lat
            )
            self._zero_offset = (
                render_config.bounding_box.bottom_left.lon,
                render_config.bounding_box.top_right.lat,
            )
            self._hide_completed_orders = render_config.hide_completed_orders

            self._pygame_window = None
            self._pygame_clock = None
            self._pygame_font = None
        elif self._render_mode == RenderMode.WEB:
            if not isinstance(self._config.render.config, WebRenderConfig):
                raise ValueError(
                    f"Web render mode requires WebRenderConfig, but got {self._config.render.config}"
                )
            self._render_fps = self._config.render.config.render_fps
            self._render_server_host = self._config.render.config.server_host
            self._pygame_clock = None

        set_global_seeds(self._config.seed)

        self._order_generator: Optional[order_generators.OrderGenerator] = None
        self._simulation_id: str = ""
        self._workers: dict[str, Worker] = {}
        self._orders: dict[str, Order] = {}
        self._service_stations: dict[str, ServiceStation] = {}
        self._rewards: dict[int, float] = {}
        self._current_time: int = 0
        self._done: bool = True
        self._actions: dict[int, Action] = {}

    @property
    def location_mode(self) -> LocationMode:
        return self._location_mode

    @property
    def step_size(self) -> int:
        return self._dt

    @property
    def routing_provider(self) -> Optional[RoutingProvider]:
        return self._routing_provider

    def get_order(self, order_id: str) -> Order:
        if order_id not in self._orders:
            raise ValueError(f"Order {order_id} does not exist")
        return self._orders[order_id]

    def get_service_station(self, service_station_id: str) -> ServiceStation:
        if service_station_id not in self._service_stations:
            raise ValueError(f"Service station {service_station_id} does not exist")
        return self._service_stations[service_station_id]

    def reset(self, simulation_id: Optional[str] = None) -> tuple[Observation, Info]:
        logger.debug("Resetting simulator")
        self._simulation_id = simulation_id if simulation_id is not None else str(uuid.uuid4())
        self._current_time = self._config.start_time
        self._done = False
        self._rewards = {}
        self._actions = {}

        self._workers = {}
        worker_colors = generate_colors(len(self._config.workers))
        for worker_config, worker_color in zip(self._config.workers, worker_colors):
            self._workers[worker_config.id] = Worker(
                id=worker_config.id,
                initial_location=worker_config.initial_location,
                initial_time=self._current_time,
                travel_type=worker_config.travel_type,
                speed=worker_config.speed,
                fuel_consumption_rate=worker_config.fuel_consumption_rate,
                color=worker_color,
                sim=self,
                logger=logger,
            )

        self._order_generator = getattr(
            order_generators, self._config.order_generator.generator_type
        )(self._config.order_generator.config)
        # TODO(dburakov): Probably, need to generate all orders in advance
        self._orders = {}
        self._generate_orders(self._current_time)

        self._service_stations = {}
        for service_station_config in self._config.service_stations:
            self._service_stations[service_station_config.id] = ServiceStation(
                id=service_station_config.id,
                location=service_station_config.location,
                service_time=service_station_config.service_time,
            )

        self._render_frame()

        return self._get_current_observation(), self._get_current_info()

    def _generate_orders(self, current_time: int) -> None:
        if self._order_generator is None:
            raise ValueError("Order generator is not set")

        for order in self._order_generator.generate(current_time):
            self._orders[order.id] = order
            self._orders[order.id].set_cancellation_threshold(
                self._config.order_cancellation_threshold  # type: ignore
            )

    def step(self, action: Action) -> tuple[Observation, float, bool, bool, Info]:
        logging.debug(f"Taking step with action: {action}")

        self._perform_action(action)
        self._update_state()
        reward = self._calculate_current_reward()

        self._render_frame()

        truncated = False
        if self._done:
            has_uncompleted_orders = any(
                order.status in {OrderStatus.COMPLETED, OrderStatus.CANCELED}
                for order in self._orders.values()
            )
            truncated = has_uncompleted_orders

        return (
            self._get_current_observation(),
            reward,
            self._done,
            truncated,
            self._get_current_info(),
        )

    def render(self) -> Optional[dict]:
        if self._render_mode == RenderMode.JSON:
            return self._render_json()

    def close(self) -> None:
        if self._render_mode == RenderMode.PYGAME:
            if self._pygame_window is not None:
                pygame.display.quit()
                pygame.quit()

    def assign_order(self, order_id: str, worker_id: str) -> None:
        self._basic_check(order_id, worker_id)
        order = self._orders[order_id]

        if order.status != OrderStatus.CREATED:
            raise ValueError(f"Cannot assign order {order_id} with status {order.status}")

        if order.assigned_worker_id is not None:
            raise ValueError(
                f"Order {order_id} is already assigned to worker {order.assigned_worker_id}"
            )

        order.assign(worker_id, self._current_time)

    def pickup_order(self, order_id: str, worker_id: str) -> int:
        self._basic_check(order_id, worker_id)
        order = self._orders[order_id]

        if order.status != OrderStatus.ASSIGNED:
            raise ValueError(f"Cannot pick up order {order_id} with status {order.status}")

        if order.assigned_worker_id != worker_id:
            raise ValueError(
                f"Order {order_id} is not assigned to worker {worker_id}. Current assignment: {order.assigned_worker_id}"
            )

        worker = self._workers[worker_id]

        if not worker.location.near(order.from_location):
            raise ValueError(
                f"Worker {worker_id} is not at the order {order.id} pickup location. Worker location: {worker.location}, order pickup location: {order.from_location}"
            )

        # TODO(dburakov): Add pickup time sampler
        pickup_time = self._config.order_pickup_time
        pickup_end_time = self._current_time + pickup_time

        order.pickup(worker_id, self._current_time, pickup_end_time)

        return pickup_end_time

    def drop_off_order(self, order_id: str, worker_id: str) -> int:
        self._basic_check(order_id, worker_id)
        order = self._orders[order_id]

        if order.status != OrderStatus.IN_DELIVERY:
            raise ValueError(f"Cannot drop off order {order_id} with status {order.status}")

        worker = self._workers[worker_id]

        if worker.id != order.assigned_worker_id:
            raise ValueError(
                f"Worker {worker_id} is not assigned to order {order.id}. Assigned worker: {order.assigned_worker_id}"
            )

        if not worker.location.near(order.to_location):
            raise ValueError(
                f"Worker {worker_id} is not at the order {order.id} drop off location. Worker location: {worker.location}, order drop off location: {order.to_location}"
            )

        # TODO(dburakov): Add drop off time sampler
        drop_off_time = self._config.order_drop_off_time
        drop_off_end_time = self._current_time + drop_off_time

        order.drop_off(self._current_time, drop_off_end_time)

        return drop_off_end_time

    def service(self, worker_id: str, service_station_id: str, max_service_time: int) -> int:
        if worker_id not in self._workers:
            raise ValueError(f"Worker {worker_id} does not exist")

        if service_station_id not in self._service_stations:
            raise ValueError(f"Service station {service_station_id} does not exist")

        worker = self._workers[worker_id]
        service_station = self._service_stations[service_station_id]

        if not worker.location.near(service_station.location):
            raise ValueError(
                f"Worker {worker_id} is not at the service station {service_station_id}. Worker location: {worker.location}, service station location: {service_station.location}"
            )

        service_end_time = service_station.service(worker_id, max_service_time, self._current_time)

        return service_end_time

    def _get_current_observation(self) -> Observation:
        workers = [worker.get_observation() for worker in self._workers.values()]
        orders = [
            order.get_observation()
            for order in self._orders.values()
            if not order.status.is_terminal()
        ]
        service_stations = [
            service_station.get_observation() for service_station in self._service_stations.values()
        ]
        observation = Observation(
            current_time=self._current_time,
            workers=workers,
            orders=orders,
            service_stations=service_stations,
        )
        return observation

    def _get_current_info(self) -> Info:
        workers = []
        orders = []
        service_stations = []
        metrics = []
        if self._done:
            worker_to_completed_orders = {worker.id: 0 for worker in self._workers.values()}
            worker_to_assigned_orders = {worker.id: 0 for worker in self._workers.values()}
            average_time_to_assign = 0.0
            average_time_to_pickup = 0.0
            n_completed_orders = 0
            n_assigned_orders = 0
            for order in self._orders.values():
                if order.status != OrderStatus.CANCELED and order.status >= OrderStatus.ASSIGNED:
                    n_assigned_orders += 1

                if order.status == OrderStatus.COMPLETED:
                    if order.pickup_start_time is None:
                        raise ValueError("Pickup start time is not set for completed order")
                    average_time_to_pickup += order.pickup_start_time - order.creation_time

                    if order.assignment_time is None:
                        raise ValueError("Assignment time is not set for completed order")
                    average_time_to_assign += order.assignment_time - order.creation_time
                    n_completed_orders += 1

                    if order.assigned_worker_id is None:
                        raise ValueError("Assigned worker is not set for completed order")
                    worker_to_completed_orders[order.assigned_worker_id] += 1

                if order.assigned_worker_id is not None:
                    worker_to_assigned_orders[order.assigned_worker_id] += 1

                orders.append(
                    OrderInfo(
                        id=order.id,
                        client_id=order.client_id,
                        from_location=order.from_location,
                        to_location=order.to_location,
                        creation_time=order.creation_time,
                        time_window=order.time_window,
                        status=order.status,
                        assignment_time=order.assignment_time,
                        pickup_start_time=order.pickup_start_time,
                        pickup_end_time=order.pickup_end_time,
                        drop_off_start_time=order.drop_off_start_time,
                        drop_off_end_time=order.drop_off_end_time,
                        completion_time=order.completion_time,
                        assigned_worker_id=order.assigned_worker_id,
                    )
                )

            if n_completed_orders != 0:
                average_time_to_pickup /= n_completed_orders
                average_time_to_assign /= n_completed_orders
            else:
                average_time_to_pickup = float("inf")
                average_time_to_assign = float("inf")

            worker_to_station_visits = {worker.id: 0 for worker in self._workers.values()}
            for service_station in self._service_stations.values():
                events = []
                for event in service_station.service_events:
                    worker_to_station_visits[event.worker_id] += 1
                    events.append((event.worker_id, event.start_time, event.end_time))

                service_stations.append(
                    ServiceStationInfo(
                        id=service_station.id,
                        location=service_station.location,
                        service_events=events,
                    )
                )

            average_traveled_distance = 0.0
            average_service_station_visits = 0.0
            average_idle_rate = 0.0
            average_with_order_rate = 0.0
            for worker in self._workers.values():
                status_history = worker.status_history
                status_history.append((self._current_time, worker.status))
                worker_total_time = status_history[-1][0] - status_history[0][0]
                worker_idle_time = 0
                worker_with_order_time = 0
                for cur_status, next_status in zip(status_history, status_history[1:]):
                    if cur_status[1] == WorkerStatus.IDLE:
                        worker_idle_time += next_status[0] - cur_status[0]
                    elif cur_status[1] in (
                        WorkerStatus.PICKING_UP,
                        WorkerStatus.MOVING_TO_DROP_OFF,
                        WorkerStatus.DROPPING_OFF,
                    ):
                        worker_with_order_time += next_status[0] - cur_status[0]
                worker_idle_rate = 100.0 * worker_idle_time / worker_total_time
                worker_with_order_rate = 100.0 * worker_with_order_time / worker_total_time

                worker_completion_rate = 0.0
                if worker_to_assigned_orders[worker.id] > 0:
                    worker_completion_rate = (
                        100.0
                        * worker_to_completed_orders[worker.id]
                        / worker_to_assigned_orders[worker.id]
                    )

                workers.append(
                    WorkerInfo(
                        id=worker.id,
                        travel_type=worker.travel_type,
                        speed=worker.speed,
                        color=worker.color,
                        status_history=status_history,
                        n_assigned_orders=worker_to_assigned_orders[worker.id],
                        n_completed_orders=worker_to_completed_orders[worker.id],
                        completion_rate=worker_completion_rate,
                        idle_rate=worker_idle_rate,
                        with_order_rate=worker_with_order_rate,
                        traveled_distance=worker.traveled_distance,
                        consumed_fuel=worker.consumed_fuel,
                        n_service_station_visits=worker_to_station_visits[worker.id],
                    )
                )

                average_service_station_visits += worker_to_station_visits[worker.id]
                average_traveled_distance += worker.traveled_distance
                average_idle_rate += worker_idle_rate
                average_with_order_rate += worker_with_order_rate

            if len(self._workers) != 0:
                average_traveled_distance /= len(self._workers)
                average_service_station_visits /= len(self._workers)
                average_idle_rate /= len(self._workers)
                average_with_order_rate /= len(self._workers)

            total_reward = sum(self._rewards.values())

            metrics.append(
                {
                    "name": "Total reward",
                    "value": total_reward,
                    "unit": "",
                }
            )
            metrics.append(
                {
                    "name": "Number of workers",
                    "value": len(workers),
                    "unit": "",
                }
            )
            metrics.append(
                {
                    "name": "Total number of orders",
                    "value": len(orders),
                    "unit": "",
                }
            )
            metrics.append(
                {
                    "name": "Number of completed orders",
                    "value": n_completed_orders,
                    "unit": "",
                }
            )
            metrics.append(
                {
                    "name": "Average time to assign",
                    "value": average_time_to_assign,
                    "unit": "steps" if self._location_mode == LocationMode.CARTESIAN else "min",
                }
            )
            metrics.append(
                {
                    "name": "Average time to pickup",
                    "value": average_time_to_pickup,
                    "unit": "steps" if self._location_mode == LocationMode.CARTESIAN else "min",
                }
            )
            metrics.append(
                {
                    "name": "Assignment rate",
                    "value": 100.0 * n_assigned_orders / len(orders),
                    "unit": "%",
                }
            )
            metrics.append(
                {
                    "name": "Completion rate",
                    "value": 100.0 * n_completed_orders / len(orders),
                    "unit": "%",
                }
            )
            metrics.append(
                {
                    "name": "Worker average idle rate",
                    "value": average_idle_rate,
                    "unit": "%",
                }
            )
            metrics.append(
                {
                    "name": "Worker average with order rate",
                    "value": average_with_order_rate,
                    "unit": "%",
                }
            )
            metrics.append(
                {
                    "name": "Worker average traveled distance",
                    "value": average_traveled_distance,
                    "unit": "" if self._location_mode == LocationMode.CARTESIAN else "km",
                }
            )
            metrics.append(
                {
                    "name": "Worker average service station visits",
                    "value": average_service_station_visits,
                    "unit": "",
                }
            )

        info = Info(
            start_time=self._config.start_time,
            end_time=self._config.end_time,
            workers=workers,
            orders=orders,
            service_stations=service_stations,
            metrics=metrics,
        )
        return info

    def _perform_action(self, action: Action) -> None:
        if self._done:
            raise ValueError("Cannot perform action when simulation is done")

        self._actions[self._current_time] = action

        for worker_action in action.worker_actions:
            if worker_action.worker_id not in self._workers:
                raise ValueError(f"Worker {worker_action.worker_id} does not exist")

            worker = self._workers[worker_action.worker_id]
            worker.perform_action(worker_action.action, self._current_time)

    def _basic_check(self, order_id: str, worker_id: str) -> None:
        if worker_id not in self._workers:
            raise ValueError(f"Worker {worker_id} does not exist")

        if order_id not in self._orders:
            raise ValueError(f"Order {order_id} does not exist")

    def _update_state(self) -> None:
        self._current_time += self._dt
        self._done = self._current_time >= self._config.end_time

        if self._done:
            return

        for order in self._orders.values():
            if order.status.is_terminal():
                continue
            order.update_state(self._current_time)

        for worker in self._workers.values():
            worker.update_state(self._current_time)

        # TODO(dburakov): Probably, need to generate all orders in advance
        self._generate_orders(self._current_time)

    def _calculate_current_reward(self) -> float:
        # TODO(dburakov): Think about proper reward function
        current_reward = 0.0
        for order in self._orders.values():
            if (
                order.status == OrderStatus.COMPLETED
                and order.completion_time is not None
                and order.completion_time == self._current_time
            ):
                # TODO(dburakov): Think about complex time windows
                if len(order.time_window) != 1:
                    raise ValueError("Only simple time windows are supported")

                start_penalty = max(0, order.time_window[0][0] - order.completion_time)
                end_penalty = max(0, order.completion_time - order.time_window[0][1])
                current_reward += max(start_penalty, end_penalty)

            if self._done and order.status != OrderStatus.COMPLETED:
                start_penalty = max(0, order.time_window[0][0] - self._current_time)
                end_penalty = max(0, self._current_time - order.time_window[0][1])
                current_reward += max(start_penalty, end_penalty)

        # Reward is negative since it is a penalty
        current_reward = -1.0 * current_reward

        self._rewards[self._current_time] = current_reward
        return current_reward

    def _render_frame(self) -> None:
        if self._render_mode == RenderMode.PYGAME:
            self._render_pygame_frame()
        elif self._render_mode == RenderMode.WEB:
            self._render_web_frame()

    def _render_pygame_frame(self) -> None:
        if self._pygame_window is None:
            pygame.init()
            self._pygame_window = pygame.display.set_mode(
                self._window_size, flags=pygame.RESIZABLE | pygame.SCALED
            )
            pygame.display.set_caption("Logistics Simulator")

        if self._pygame_clock is None:
            self._pygame_clock = pygame.time.Clock()

        if self._pygame_font is None:
            self._pygame_font = pygame.font.Font(None, 20)

        surface = pygame.Surface(self._window_size)
        surface.fill((255, 255, 255))

        for service_station in self._service_stations.values():
            pygame.draw.circle(
                surface=surface,
                color=(0, 0, 0),
                center=self._location_to_coordinates(service_station.location),
                radius=8,
            )
            pygame.draw.circle(
                surface=surface,
                color=(150, 0, 245),
                center=self._location_to_coordinates(service_station.location),
                radius=4,
            )

        for order in self._orders.values():
            if self._hide_completed_orders and order.status.is_terminal():
                continue

            if order.status < OrderStatus.IN_DELIVERY:
                color = (0, 0, 255)
                location = order.from_location
            else:
                color = (0, 255, 0)
                location = order.to_location

            pygame.draw.circle(
                surface=surface,
                color=color,
                center=self._location_to_coordinates(location),
                radius=3,
            )

        for worker in self._workers.values():
            pygame.draw.circle(
                surface=surface,
                color=worker.color,
                center=self._location_to_coordinates(worker.location),
                width=3,
                radius=5,
            )

            if WorkerStatus.is_moving_status(worker.status):
                if worker.busy_until is None:
                    raise ValueError("Worker is moving, but busy_until is not set")

                if worker.path is None:
                    raise ValueError("Worker is moving, but path is not set")

                pygame.draw.line(
                    surface,
                    worker.color,
                    self._location_to_coordinates(worker.location),
                    self._location_to_coordinates(worker.path[worker.busy_until]),
                    width=2,
                )

        current_time = self._pygame_font.render(
            f"Current time: {self._current_time}", True, (0, 0, 0)
        )

        self._pygame_window.blit(surface, surface.get_rect())
        self._pygame_window.blit(current_time, (10, 10))
        pygame.event.pump()
        pygame.display.update()

        self._pygame_clock.tick(self._render_fps)

    def _render_web_frame(self) -> None:
        if self._pygame_clock is None:
            self._pygame_clock = pygame.time.Clock()

        json_data = self._render_json()
        response = requests.post(
            f"http://{self._render_server_host}/render", json=json_data, timeout=1.0
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to render web frame. Response status code: {response.status_code} Response body: {response.text}"
            )

        self._pygame_clock.tick(self._render_fps)

    def _location_to_coordinates(self, location: Location) -> tuple[int, int]:
        return (
            int((location.lon - self._zero_offset[0]) * self._x_scale),
            int((self._zero_offset[1] - location.lat) * self._y_scale),
        )

    def _render_json(self) -> dict:
        bounds = {"min": {"lat": 90.0, "lon": 180.0}, "max": {"lat": -90.0, "lon": -180.0}}
        json_workers = []
        for worker in self._workers.values():
            self._update_bounds(bounds, worker.location)
            encoded_path = None
            remaining_path_index = None
            worker_path = worker.path
            if worker_path is not None:
                if worker.route is not None:
                    encoded_path = worker.route.geometry
                    remaining_path_index = worker.remaining_path_index
                else:
                    raw_path = []
                    for i, (time, location) in enumerate(worker_path.items()):
                        self._update_bounds(bounds, location)
                        raw_path.append((location.lat, location.lon))
                        if remaining_path_index is None and time >= self._current_time:
                            remaining_path_index = i
                    encoded_path = polyline.encode(raw_path)

            json_workers.append(
                {
                    "id": worker.id,
                    "location": worker.location.to_dict(),
                    "travel_type": worker.travel_type.value,
                    "speed": worker.speed,
                    "fuel": worker.fuel,
                    "path": encoded_path,
                    "remaining_path_index": remaining_path_index,
                    "status": worker.status.value,
                    "color": worker.color,
                }
            )

        json_orders = []
        n_active_orders = 0
        n_completed_orders = 0
        n_canceled_orders = 0
        average_time_to_assign = 0.0
        average_time_to_pickup = 0.0
        for order in self._orders.values():
            if order.status == OrderStatus.COMPLETED:
                if order.pickup_start_time is None:
                    raise ValueError("Pickup start time is not set for completed order")
                average_time_to_pickup += order.pickup_start_time - order.creation_time

                if order.assignment_time is None:
                    raise ValueError("Assignment time is not set for completed order")
                average_time_to_assign += order.assignment_time - order.creation_time

                n_completed_orders += 1
                continue
            elif order.status == OrderStatus.CANCELED:
                n_canceled_orders += 1
                continue

            n_active_orders += 1

            self._update_bounds(bounds, order.from_location)
            self._update_bounds(bounds, order.to_location)
            json_orders.append(
                {
                    "id": order.id,
                    "client_id": order.client_id,
                    "from_location": order.from_location.to_dict(),
                    "to_location": order.to_location.to_dict(),
                    "creation_time": order.creation_time,
                    "time_window": order.time_window,
                    "status": order.status.value,
                }
            )
        if n_completed_orders != 0:
            average_time_to_pickup /= n_completed_orders
            average_time_to_assign /= n_completed_orders

        json_service_stations = []
        for service_station in self._service_stations.values():
            self._update_bounds(bounds, service_station.location)
            json_service_stations.append(
                {
                    "id": service_station.id,
                    "location": service_station.location.to_dict(),
                }
            )

        json_observation = {
            "current_time": self._current_time,
            "bounds": bounds,
            "workers": json_workers,
            "orders": json_orders,
            "service_stations": json_service_stations,
        }
        json_info = {
            "simulation_id": self._simulation_id,
            "start_time": self._config.start_time,
            "end_time": self._config.end_time,
            "metrics": [
                {
                    "name": "Number of workers",
                    "value": len(self._workers),
                    "unit": "",
                },
                {
                    "name": "Number of active orders",
                    "value": n_active_orders,
                    "unit": "",
                },
                {
                    "name": "Number of completed orders",
                    "value": n_completed_orders,
                    "unit": "",
                },
                {
                    "name": "Number of canceled orders",
                    "value": n_canceled_orders,
                    "unit": "",
                },
                {
                    "name": "Average time to assign",
                    "value": average_time_to_assign,
                    "unit": "steps" if self._location_mode == LocationMode.CARTESIAN else "min",
                },
                {
                    "name": "Average time to pickup",
                    "value": average_time_to_pickup,
                    "unit": "steps" if self._location_mode == LocationMode.CARTESIAN else "min",
                },
            ],
        }

        json_data = {
            "observation": json_observation,
            "info": json_info,
        }

        return json_data

    def _update_bounds(self, bounds: dict, location: Location) -> None:
        bounds["min"]["lat"] = min(bounds["min"]["lat"], location.lat)
        bounds["min"]["lon"] = min(bounds["min"]["lon"], location.lon)
        bounds["max"]["lat"] = max(bounds["max"]["lat"], location.lat)
        bounds["max"]["lon"] = max(bounds["max"]["lon"], location.lon)

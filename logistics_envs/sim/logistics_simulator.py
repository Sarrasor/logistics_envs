import logging
from typing import Optional

import pygame

from logistics_envs.sim import order_generators
from logistics_envs.sim.structs.action import Action
from logistics_envs.sim.structs.common import Location
from logistics_envs.sim.structs.config import (
    LocationMode,
    LogisticsSimulatorConfig,
    PygameRenderConfig,
    RenderMode,
)
from logistics_envs.sim.structs.info import Info
from logistics_envs.sim.structs.observation import Observation
from logistics_envs.sim.structs.order import Order, OrderStatus
from logistics_envs.sim.structs.worker import Worker, WorkerStatus
from logistics_envs.sim.utils import generate_colors, set_global_seeds

logger = logging.getLogger("logistics_simulator")


class LogisticsSimulator:
    def __init__(self, config: LogisticsSimulatorConfig) -> None:
        self._config = config
        self._location_mode = self._config.location_mode
        self._dt = self._config.step_size
        self._render_mode = self._config.render.render_mode
        if self._render_mode == RenderMode.PYGAME:
            if self._config.render.config is None:
                raise ValueError("Pygame render mode requires render config")

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

        set_global_seeds(self._config.seed)

        self._order_generator = getattr(
            order_generators, self._config.order_generator.generator_type
        )(self._config.order_generator.config)

        self._workers: dict[str, Worker] = {}
        self._orders: dict[str, Order] = {}
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

    def get_order(self, order_id: str) -> Order:
        if order_id not in self._orders:
            raise ValueError(f"Order {order_id} does not exist")
        return self._orders[order_id]

    def reset(self) -> tuple[Observation, Info]:
        logger.debug("Resetting simulator")

        self._current_time = self._config.start_time
        self._done = False

        self._actions = {}

        self._workers = {}
        worker_colors = generate_colors(len(self._config.workers))
        for worker_config, worker_color in zip(self._config.workers, worker_colors):
            self._workers[worker_config.id] = Worker(
                id=worker_config.id,
                initial_location=worker_config.initial_location,
                travel_type=worker_config.travel_type,
                speed=worker_config.speed,
                color=worker_color,
                sim=self,
                logger=logger,
            )

        # TODO(dburakov): Probably, need to generate all orders in advance
        self._orders = {}
        for order in self._order_generator.generate(self._current_time):
            self._orders[order.id] = order

        if self._render_mode == RenderMode.PYGAME:
            self._render_frame()

        return self._get_current_observation(), self._get_current_info()

    def step(self, action: Action) -> tuple[Observation, float, bool, bool, Info]:
        logging.debug(f"Taking step with action: {action}")

        self._perform_action(action)
        self._update_state()
        reward = self._calculate_current_reward()

        if self._render_mode == RenderMode.PYGAME:
            self._render_frame()

        return (
            self._get_current_observation(),
            reward,
            self._done,
            False,  # For now assume no truncation
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

    def pickup_order(self, order_id: str, worker_id: str) -> int:
        self._basic_check(order_id, worker_id)
        order = self._orders[order_id]

        if order.status != OrderStatus.CREATED:
            raise ValueError(f"Cannot pick up order {order_id} with status {order.status}")

        if order.assigned_worker_id is not None:
            raise ValueError(
                f"Order {order_id} is already assigned to worker {order.assigned_worker_id}"
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

    def _get_current_observation(self) -> Observation:
        workers = [worker.get_observation() for worker in self._workers.values()]
        orders = [
            order.get_observation()
            for order in self._orders.values()
            if order.status != OrderStatus.COMPLETED
        ]
        observation = Observation(current_time=self._current_time, workers=workers, orders=orders)
        return observation

    def _get_current_info(self) -> Info:
        info = Info()
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
            order.update_state(self._current_time)

        for worker in self._workers.values():
            worker.update_state(self._current_time)

        # TODO(dburakov): Probably, need to generate all orders in advance
        for order in self._order_generator.generate(self._current_time):
            self._orders[order.id] = order

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
                current_reward += 1e6

        # Reward is negative since it is a penalty
        current_reward = -1.0 * current_reward

        self._rewards[self._current_time] = current_reward
        return current_reward

    def _render_frame(self) -> None:
        if self._render_mode != RenderMode.PYGAME:
            return

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

        for worker in self._workers.values():
            pygame.draw.circle(
                surface=surface,
                color=worker.color,
                center=self._location_to_coordinates(worker.location),
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

        for order in self._orders.values():
            if self._hide_completed_orders and order.status == OrderStatus.COMPLETED:
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

        current_time = self._pygame_font.render(
            f"Current time: {self._current_time}", True, (0, 0, 0)
        )

        self._pygame_window.blit(surface, surface.get_rect())
        self._pygame_window.blit(current_time, (10, 10))
        pygame.event.pump()
        pygame.display.update()

        self._pygame_clock.tick(self._render_fps)

    def _location_to_coordinates(self, location: Location) -> tuple[int, int]:
        return (
            int((location.lon - self._zero_offset[0]) * self._x_scale),
            int((self._zero_offset[1] - location.lat) * self._y_scale),
        )

    def _render_json(self) -> dict:
        raise NotImplementedError("JSON render mode is not implemented yet")

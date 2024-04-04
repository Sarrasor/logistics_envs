import logging
from dataclasses import asdict

from logistics_envs.sim import order_generators
from logistics_envs.sim.structs.action import Action
from logistics_envs.sim.structs.config import LogisticsSimulatorConfig
from logistics_envs.sim.structs.info import Info
from logistics_envs.sim.structs.observation import Observation
from logistics_envs.sim.structs.order import Order
from logistics_envs.sim.structs.worker import Worker
from logistics_envs.sim.utils import generate_colors

logger = logging.getLogger("logistics_simulator")


class LogisticsSimulator:
    def __init__(self, config: LogisticsSimulatorConfig) -> None:
        self._config = config
        self._location_mode = self._config.location_mode
        self._dt = self._config.step_size

        self._order_generator = getattr(
            order_generators, self._config.order_generator.generator_type
        )(self._config.order_generator.config)

        self._workers: dict[str, Worker] = {}
        self._orders: dict[str, Order] = {}
        self._current_time: int = 0
        self._done: bool = True

    def reset(self) -> tuple[Observation, Info]:
        logger.debug("Resetting simulator")

        self._current_time = self._config.start_time
        self._done = False

        self._workers = {}
        worker_colors = generate_colors(len(self._config.workers))
        for worker_config, worker_color in zip(self._config.workers, worker_colors):
            self._workers[worker_config.id] = Worker(
                **asdict(worker_config),
                color=worker_color,
                dt=self._dt,
                location_mode=self._location_mode,
                logger=logger,
            )

        self._orders = {}
        for order in self._order_generator.generate(self._current_time):
            self._orders[order.id] = order

        return self._get_current_observation(), self._get_current_info()

    def step(self, action: Action) -> tuple[Observation, float, bool, bool, Info]:
        current_reward = self._perform_action(action)
        self._current_time += self._dt
        self._done = self._current_time >= self._config.end_time

        return (
            self._get_current_observation(),
            current_reward,
            self._done,
            False,  # For now assume no truncation
            self._get_current_info(),
        )

    def close(self) -> None:
        pass

    def _get_current_observation(self) -> Observation:
        workers = [worker.get_observation() for worker in self._workers.values()]
        orders = [order.get_observation() for order in self._orders.values()]
        observation = Observation(current_time=self._current_time, workers=workers, orders=orders)
        return observation

    def _get_current_info(self) -> Info:
        info = Info()
        return info

    def _perform_action(self, action: Action) -> float:
        if self._done:
            raise ValueError("Cannot perform action when simulation is done")

        return 0.0

    def _render_frame(self) -> None:
        pass

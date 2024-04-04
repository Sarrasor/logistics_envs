from collections import defaultdict
from dataclasses import asdict
from logistics_envs.sim.order_generators.order_generator import OrderGenerator
from logistics_envs.sim.structs.config import PredefinedOrderGeneratorConfig
from logistics_envs.sim.structs.order import Order


class PredefinedOrderGenerator(OrderGenerator):
    def __init__(self, config: PredefinedOrderGeneratorConfig) -> None:
        self._config = config
        self._time_to_order_list = defaultdict(list)

        for order_config in self._config.orders:
            # TODO(dburakov): Make sure request time matches with time discretization
            self._time_to_order_list[order_config.creation_time].append(
                Order(**asdict(order_config))
            )

    def generate(self, current_time: int) -> list[Order]:
        return self._time_to_order_list[current_time]

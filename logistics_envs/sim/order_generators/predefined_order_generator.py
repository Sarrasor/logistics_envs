from collections import defaultdict
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
                Order(
                    id=order_config.id,
                    client_id=order_config.client_id,
                    from_location=order_config.from_location,
                    to_location=order_config.to_location,
                    creation_time=order_config.creation_time,
                    time_window=order_config.time_window,
                )
            )

    def generate(self, current_time: int) -> list[Order]:
        return self._time_to_order_list[current_time]

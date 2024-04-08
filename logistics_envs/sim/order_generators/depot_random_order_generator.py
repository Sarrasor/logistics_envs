import numpy as np
from logistics_envs.sim.order_generators.order_generator import OrderGenerator
from logistics_envs.sim.structs.common import Location
from logistics_envs.sim.structs.config import DepotRandomOrderGeneratorConfig
from logistics_envs.sim.structs.order import Order


class DepotRandomOrderGenerator(OrderGenerator):
    def __init__(self, config: DepotRandomOrderGeneratorConfig) -> None:
        self._config = config
        if self._config.max_concurrent_orders < 1:
            raise ValueError(
                f"max_concurrent_orders={self._config.max_concurrent_orders} must be greater than 0"
            )

    def generate(self, current_time: int) -> list[Order]:
        orders = []

        if not (
            self._config.order_generation_start_time
            <= current_time
            <= self._config.order_generation_end_time
        ):
            return orders

        if np.random.random() > self._config.generation_probability:
            return orders

        if self._config.max_concurrent_orders == 1:
            n_orders = 1
        else:
            n_orders = np.random.randint(1, self._config.max_concurrent_orders)
        for i in range(n_orders):
            order_location = Location(
                lat=np.random.uniform(*self._config.lat_range),
                lon=np.random.uniform(*self._config.lon_range),
            )

            orders.append(
                Order(
                    id=f"random_order_{current_time}_{i}",
                    client_id="random_client",
                    from_location=self._config.depot_location,
                    to_location=order_location,
                    creation_time=current_time,
                    time_window=[(current_time, current_time + self._config.window_size)],
                )
            )

        return orders

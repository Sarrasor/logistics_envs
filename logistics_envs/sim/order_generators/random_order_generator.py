import numpy as np
from logistics_envs.sim.order_generators.order_generator import OrderGenerator
from logistics_envs.sim.structs.common import Location
from logistics_envs.sim.structs.config import RandomOrderGeneratorConfig
from logistics_envs.sim.structs.order import Order


class RandomOrderGenerator(OrderGenerator):
    def __init__(self, config: RandomOrderGeneratorConfig) -> None:
        self._config = config
        if self._config.max_concurrent_orders < 1:
            raise ValueError(
                f"max_concurrent_orders={self._config.max_concurrent_orders} must be greater than 0"
            )

    def generate(self, current_time: int) -> list[Order]:
        orders = []

        if np.random.random() > self._config.generation_probability:
            return orders

        n_orders = np.random.randint(1, self._config.max_concurrent_orders)
        for i in range(n_orders):
            from_location = np.array(
                [
                    np.random.uniform(*self._config.lat_range),
                    np.random.uniform(*self._config.lon_range),
                ]
            )
            to_location = np.array(
                [
                    np.random.uniform(*self._config.lat_range),
                    np.random.uniform(*self._config.lon_range),
                ]
            )

            if not self._config.allow_same_location:
                while np.all(from_location == to_location):
                    to_location = np.array(
                        [
                            np.random.uniform(*self._config.lat_range),
                            np.random.uniform(*self._config.lon_range),
                        ]
                    )

            orders.append(
                Order(
                    id=f"random_order_{current_time}_{i}",
                    client_id="random_client",
                    from_location=Location(lat=from_location[0], lon=from_location[1]),
                    to_location=Location(lat=to_location[0], lon=to_location[1]),
                    creation_time=current_time,
                    time_window=[(current_time, current_time + self._config.window_size)],
                )
            )

        return orders

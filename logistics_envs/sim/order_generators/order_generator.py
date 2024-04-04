from abc import ABC, abstractmethod
from typing import List

from logistics_envs.sim.structs.order import Order


class OrderGenerator(ABC):
    @abstractmethod
    def generate(self, current_time: int) -> List[Order]:
        pass

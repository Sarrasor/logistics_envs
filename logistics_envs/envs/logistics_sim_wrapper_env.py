from abc import abstractmethod
from typing import Optional
import gymnasium as gym

from logistics_envs.sim.logistics_simulator import LogisticsSimulator
from logistics_envs.sim.structs.action import Action
from logistics_envs.sim.structs.config import LogisticsSimulatorConfig
from logistics_envs.sim.structs.info import Info
from logistics_envs.sim.structs.observation import Observation


class LogisticsSimWrapperEnv(gym.Env):
    def __init__(self, config: LogisticsSimulatorConfig) -> None:
        self._sim = LogisticsSimulator(config)

    def reset(
        self,
        *,
        seed=None,
        options: Optional[dict] = None,
    ) -> tuple[dict, dict]:
        if not options:
            options = {}
        sim_observation, sim_info = self._sim.reset(**options)
        observation = self._convert_to_observation(sim_observation)
        info = self._convert_to_info(sim_info)

        return observation, info

    def step(self, action: dict) -> tuple[dict, float, bool, bool, dict]:
        sim_action = self._convert_to_sim_action(action)
        sim_observation, reward, done, truncated, sim_info = self._sim.step(sim_action)
        observation = self._convert_to_observation(sim_observation)
        info = self._convert_to_info(sim_info)

        return observation, reward, done, truncated, info

    def render(self) -> Optional[dict]:
        return self._sim.render()

    def close(self):
        self._sim.close()

    @abstractmethod
    def _convert_to_observation(self, sim_observation: Observation) -> dict:
        pass

    @abstractmethod
    def _convert_to_info(self, sim_info: Info) -> dict:
        pass

    @abstractmethod
    def _convert_to_sim_action(self, action_dict: dict) -> Action:
        pass

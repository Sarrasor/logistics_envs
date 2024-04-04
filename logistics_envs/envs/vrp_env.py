import gymnasium as gym
from gymnasium import spaces
import numpy as np


class VRPEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, max_couriers: int, max_nodes: int):
        self._max_couriers = max_couriers
        self._max_nodes = max_nodes
        self._max_steps = self._max_couriers * self._max_nodes
        self._cur_step = 0

        self.observation_space = spaces.Dict(
            {
                "nodes": spaces.Sequence(
                    spaces.Dict(
                        {
                            "location": spaces.Box(low=0, high=100, shape=(2,), dtype=np.float32),
                            "demand": spaces.Box(low=0, high=10, shape=(1,), dtype=np.int32),
                        }
                    )
                ),
                "couriers": spaces.Sequence(
                    spaces.Dict(
                        {
                            "location": spaces.Box(low=0, high=100, shape=(2,), dtype=np.float32),
                            "capacity": spaces.Box(low=0, high=10, shape=(1,), dtype=np.int32),
                        }
                    )
                ),
            }
        )
        self.action_space = spaces.MultiDiscrete([self._max_couriers, self._max_nodes])

    def reset(self, *, seed=None, options=None) -> tuple[dict, dict]:
        super().reset(seed=seed)

        self._cur_step = 0

        observation = self._get_observation()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action: tuple[int, int]) -> tuple[dict, float, bool, bool, dict]:
        self._cur_step += 1

        observation = self._get_observation()
        reward = 0.0
        done = self._cur_step >= self._max_steps
        truncated = False
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, done, truncated, info

    def render(self):
        if self.render_mode == "human":
            self._render_frame()

    def close(self):
        pass

    def _get_observation(self):
        return self.observation_space.sample()

    def _get_info(self):
        return {"time": 0}

    def _render_frame(self):
        pass

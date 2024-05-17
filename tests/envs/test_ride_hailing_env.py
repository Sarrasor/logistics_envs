import pytest
import gymnasium as gym

from logistics_envs.envs import RideHailingEnv
from logistics_envs.sim import LocationMode


@pytest.fixture
def env() -> RideHailingEnv:
    config = {
        "mode": LocationMode.GEOGRAPHIC,
        "start_time": 0,
        "end_time": 30,
        "time_step": 1,
        "n_drivers": 3,
        "max_orders": 4,
        "order_data_path": "test_data/ride_hailing/ride_hailing_example.csv",
        "order_pickup_time": 1,
        "order_drop_off_time": 1,
        "routing_host": "localhost:8002",
        "seed": 42,
    }

    env = gym.make(
        "logistics_envs/RideHailing-v0",
        render_mode=None,
        **config,
    )
    return env


def test_reset(env: RideHailingEnv) -> None:
    observation, info = env.reset()

    assert isinstance(observation, dict)
    assert isinstance(info, dict)

    assert len(observation["drivers_status"]) == 3
    assert observation["drivers_location"].shape == (3, 2)

    assert len(observation["orders_status"]) == 4
    assert observation["orders_from_location"].shape == (4, 2)
    assert observation["orders_to_location"].shape == (4, 2)


def get_idle_action(observation: dict) -> dict:
    n_drivers = len(observation["drivers_status"])
    action = {
        "action": [0] * n_drivers,
        "target": [0] * n_drivers,
        "location": [0.0, 0.0] * n_drivers,
    }
    return action


def test_idle_action(env: RideHailingEnv) -> None:
    observation, info = env.reset()
    last_observation = observation
    done = False

    while not done:
        action = get_idle_action(observation)
        observation, reward, done, truncated, info = env.step(action)
        last_observation = observation

    env.close()

    for courier_index in range(3):
        assert last_observation["drivers_status"][courier_index] == 0

    for order_index in range(4):
        assert observation["orders_status"][order_index] == 0

    assert len(info["orders"]) == 19
    assert len(info["workers"]) == 3

    assert info["metrics"][0]["value"] == 3.0
    assert info["metrics"][1]["value"] == 19.0
    assert info["metrics"][2]["value"] == 0.0
    assert info["metrics"][3]["value"] == 0.0
    assert info["metrics"][4]["value"] == 0.0


def get_fifo_deliver_action(observation: dict) -> dict:
    n_drivers = len(observation["drivers_status"])
    action = {
        "action": [0] * n_drivers,
        "target": [0] * n_drivers,
        "location": [0.0, 0.0] * n_drivers,
    }

    assigned_orders = set()
    n_orders = observation["n_orders"]
    for driver_index in range(n_drivers):
        if observation["drivers_status"][driver_index] == 0:
            for order_index in range(n_orders):
                if (
                    observation["orders_status"][order_index] == 0
                    and order_index not in assigned_orders
                ):
                    action["action"][driver_index] = 2
                    action["target"][driver_index] = order_index
                    assigned_orders.add(order_index)
                    break

    return action


def test_deliver(env: RideHailingEnv) -> None:
    total_orders = 0
    observation, info = env.reset()
    total_orders += observation["n_orders"]
    done = False

    while not done:
        action = get_fifo_deliver_action(observation)
        observation, reward, done, truncated, info = env.step(action)
        total_orders += observation["n_orders"]

    env.close()

    assert total_orders != 0

    assert info["metrics"][0]["value"] == 3.0
    assert info["metrics"][1]["value"] == 19.0
    assert info["metrics"][2]["value"] == 2.0
    assert info["metrics"][3]["value"] == 0.0
    assert info["metrics"][4]["value"] == 10.0

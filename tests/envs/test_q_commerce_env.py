import numpy as np
import pytest
import gymnasium as gym

from logistics_envs.envs.q_commerce_env import QCommerceEnv


@pytest.fixture
def env() -> QCommerceEnv:
    config = {
        "start_time": 100,
        "end_time": 300,
        "time_step": 1,
        "depot_location": [0.3, 0.5],
        "n_couriers": 3,
        "courier_speed": 0.05,
        "max_orders": 10,
        "order_window_size": 10,
        "order_pickup_time": 2,
        "order_drop_off_time": 3,
        "order_generation_start_time": 100,
        "order_generation_end_time": 110,
        "order_generation_probability": 0.25,
        "max_concurrent_orders": 1,
        "seed": 42,
    }
    env = gym.make(
        "logistics_envs/QCommerce-v0",
        render_mode=None,
        **config,
    )
    return env


def test_reset(env: QCommerceEnv) -> None:
    observation, info = env.reset()

    assert isinstance(observation, dict)
    assert isinstance(info, dict)

    assert len(observation["couriers_status"]) == 3
    assert observation["couriers_location"].shape == (3, 2)
    assert np.allclose(observation["couriers_location"][0], [0.3, 0.5])

    assert observation["depot_location"].shape == (2,)
    assert np.allclose(observation["depot_location"], [0.3, 0.5])

    assert len(observation["orders_status"]) == 10
    assert observation["orders_from_location"].shape == (10, 2)
    assert observation["orders_to_location"].shape == (10, 2)


def get_idle_action(observation: dict) -> dict:
    n_couriers = len(observation["couriers_status"])
    action = {
        "action": [0] * n_couriers,
        "target": [0] * n_couriers,
        "location": [0.0, 0.0] * n_couriers,
    }
    return action


def test_idle_action(env: QCommerceEnv) -> None:
    observation, info = env.reset()
    last_observation = observation
    done = False

    while not done:
        action = get_idle_action(observation)
        observation, reward, done, truncated, info = env.step(action)
        last_observation = observation

    env.close()

    for courier_index in range(3):
        assert np.allclose(last_observation["couriers_location"][courier_index], [0.3, 0.5])
        assert last_observation["couriers_status"][courier_index] == 0

    for order_index in range(10):
        assert observation["orders_status"][order_index] == 0


def get_move_action(observation: dict) -> dict:
    n_couriers = len(observation["couriers_status"])
    action = {
        "action": [0] * n_couriers,
        "target": [0] * n_couriers,
        "location": [0.0, 0.0] * n_couriers,
    }

    for i in range(n_couriers):
        if observation["couriers_status"][i] == 0:
            action["action"][i] = 1
            action["location"][i] = [0.1, 0.2]

    return action


def test_move_action(env: QCommerceEnv) -> None:
    observation, info = env.reset()
    last_observation = observation
    done = False

    while not done:
        action = get_move_action(observation)
        observation, reward, done, truncated, info = env.step(action)
        last_observation = observation

    env.close()

    for courier_index in range(3):
        assert np.allclose(last_observation["couriers_location"][courier_index], [0.1, 0.2])


def get_fifo_deliver_action(observation: dict, info: dict, assigned_orders: set) -> dict:
    n_couriers = len(observation["couriers_status"])
    action = {
        "action": [0] * n_couriers,
        "target": [0] * n_couriers,
        "location": [0.0, 0.0] * n_couriers,
    }

    n_orders = observation["n_orders"]
    for courier_index in range(n_couriers):
        if observation["couriers_status"][courier_index] == 0:
            for order_index in range(n_orders):
                order_id = info["order_index_to_id"][order_index]
                if (
                    observation["orders_status"][order_index] == 0
                    and order_id not in assigned_orders
                ):
                    action["action"][courier_index] = 2
                    action["target"][courier_index] = order_index
                    assigned_orders.add(order_id)
                    break

    return action


def test_deliver(env: QCommerceEnv) -> None:
    total_orders = 0
    observation, info = env.reset()
    total_orders += observation["n_orders"]
    done = False

    assigned_orders = set()
    while not done:
        action = get_fifo_deliver_action(observation, info, assigned_orders)
        observation, reward, done, truncated, info = env.step(action)
        total_orders += observation["n_orders"]

    env.close()

    assert total_orders != 0
    assert observation["n_orders"] == 0

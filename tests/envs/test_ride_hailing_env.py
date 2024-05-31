import pytest
import gymnasium as gym

from logistics_envs.envs import RideHailingEnv
from logistics_envs.sim import LocationMode
from logistics_envs.envs.ride_hailing_env import RideHailingEnvConfig


@pytest.fixture
def env() -> RideHailingEnv:
    config = RideHailingEnvConfig.from_file(
        file_path="test_data/ride_hailing/test_cartesian.xlsx",
        mode=LocationMode.CARTESIAN,
        routing_host=None,
        render_mode=None,
        render_host=None,
    )

    env = gym.make(
        "logistics_envs/RideHailing-v0",
        config=config,
    )
    return env  # type: ignore


def test_reset(env: RideHailingEnv) -> None:
    observation, info = env.reset()

    assert isinstance(observation, dict)
    assert isinstance(info, dict)

    assert len(observation["drivers_status"]) == 3
    assert observation["drivers_location"].shape == (3, 2)

    assert len(observation["rides_status"]) == 4
    assert observation["rides_from_location"].shape == (4, 2)
    assert observation["rides_to_location"].shape == (4, 2)
    assert observation["charging_stations_location"].shape == (2, 2)


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
        assert observation["rides_status"][order_index] == 0

    assert len(info["orders"]) == 5
    assert len(info["workers"]) == 3

    assert info["metrics"][0]["value"] == -250.0
    assert info["metrics"][1]["value"] == 3.0
    assert info["metrics"][2]["value"] == 5.0
    assert info["metrics"][3]["value"] == 0.0
    assert info["metrics"][4]["value"] == float("inf")
    assert info["metrics"][5]["value"] == float("inf")
    assert info["metrics"][6]["value"] == float("inf")
    assert info["metrics"][7]["value"] == float("inf")
    assert info["metrics"][8]["value"] == 0.0
    assert info["metrics"][9]["value"] == 0.0
    assert info["metrics"][10]["value"] == 100.0
    assert info["metrics"][11]["value"] == 0.0
    assert info["metrics"][12]["value"] == 0.0
    assert info["metrics"][13]["value"] == 0.0


def get_fifo_deliver_action(observation: dict) -> dict:
    n_drivers = len(observation["drivers_status"])
    action = {
        "action": [0] * n_drivers,
        "target": [0] * n_drivers,
        "location": [0.0, 0.0] * n_drivers,
    }

    assigned_rides = set()
    n_rides = observation["n_rides"]
    for driver_index in range(n_drivers):
        if observation["drivers_status"][driver_index] == 0:
            for ride_index in range(n_rides):
                if (
                    observation["rides_status"][ride_index] == 0
                    and ride_index not in assigned_rides
                ):
                    action["action"][driver_index] = 2
                    action["target"][driver_index] = ride_index
                    assigned_rides.add(ride_index)
                    break

    return action


def test_deliver(env: RideHailingEnv) -> None:
    total_orders = 0
    observation, info = env.reset()
    total_orders += observation["n_rides"]
    done = False

    while not done:
        action = get_fifo_deliver_action(observation)
        observation, reward, done, truncated, info = env.step(action)
        total_orders += observation["n_rides"]

    env.close()

    assert total_orders != 0

    assert info["metrics"][0]["value"] == -40.0
    assert info["metrics"][1]["value"] == 3.0
    assert info["metrics"][2]["value"] == 5.0
    assert info["metrics"][3]["value"] == 5.0
    assert info["metrics"][4]["value"] == 2.6
    assert info["metrics"][5]["value"] == 8.0
    assert info["metrics"][6]["value"] == 8.0
    assert info["metrics"][7]["value"] == 6.6
    assert info["metrics"][8]["value"] == 100.0
    assert info["metrics"][9]["value"] == 100.0
    assert info["metrics"][10]["value"] == 50.0
    assert info["metrics"][11]["value"] == 32.0
    assert info["metrics"][12]["value"] > 0.0
    assert info["metrics"][13]["value"] == 0.0

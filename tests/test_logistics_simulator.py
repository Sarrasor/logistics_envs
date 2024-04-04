from logistics_envs.sim.structs.action import Action, ActionType, SpecificWorkerAction, WorkerAction
from logistics_envs.sim.structs.common import (
    Location,
)
from logistics_envs.sim.logistics_simulator import LogisticsSimulator
from logistics_envs.sim.structs.config import (
    LocationMode,
    LogisticsSimulatorConfig,
    OrderConfig,
    OrderGeneratorConfig,
    PredefinedOrderGeneratorConfig,
    WorkerConfig,
)
from logistics_envs.sim.structs.observation import Observation
from logistics_envs.sim.structs.worker import WorkerTravelType


def get_idle_action(observation: Observation) -> Action:
    worker_actions = []
    for worker_observation in observation.workers:
        worker_actions.append(
            SpecificWorkerAction(
                worker_id=worker_observation.id,
                action=WorkerAction(type=ActionType.IDLE, parameters=None),
            )
        )
    return Action(worker_actions=worker_actions)


def test_logistics_simulator():
    workers = [
        WorkerConfig(
            id="w_1", initial_location=Location(lat=0.0, lon=0.0), travel_type=WorkerTravelType.WALK
        ),
        WorkerConfig(
            id="w_2", initial_location=Location(lat=0.0, lon=0.0), travel_type=WorkerTravelType.WALK
        ),
    ]

    orders = [
        OrderConfig(
            id="o_1",
            client_id="c_1",
            from_location=Location(lat=1.0, lon=1.0),
            to_location=Location(lat=3.0, lon=1.0),
            creation_time=0,
            time_window=[(0, 100)],
        ),
        OrderConfig(
            id="o_2",
            client_id="c_2",
            from_location=Location(lat=4.0, lon=4.0),
            to_location=Location(lat=2.0, lon=3.0),
            creation_time=10,
            time_window=[(0, 100)],
        ),
    ]

    order_generator_config = OrderGeneratorConfig(
        "PredefinedOrderGenerator", PredefinedOrderGeneratorConfig(orders=orders)
    )

    config = LogisticsSimulatorConfig(
        location_mode=LocationMode.CARTESIAN,
        workers=workers,
        order_generator=order_generator_config,
        start_time=0,
        end_time=100,
        step_size=1,
    )
    sim = LogisticsSimulator(config)

    obs, info = sim.reset()
    assert isinstance(obs, Observation)

    done = False

    while not done:
        action = get_idle_action(obs)
        obs, reward, done, truncated, info = sim.step(action)
        print(
            f"Observation: {obs} Reward: {reward} Done: {done} Truncated: {truncated} Info: {info}"
        )
    sim.close()


if __name__ == "__main__":
    test_logistics_simulator()

import pytest

from logistics_envs.sim.structs.action import (
    Action,
    ActionType,
    DeliverActionParameters,
    MoveActionParameters,
    SpecificWorkerAction,
    WorkerAction,
)
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
    RenderConfig,
    RenderMode,
    WorkerConfig,
)
from logistics_envs.sim.structs.info import Info
from logistics_envs.sim.structs.observation import Observation
from logistics_envs.sim.structs.order import OrderStatus
from logistics_envs.sim.structs.worker import WorkerStatus, WorkerTravelType


@pytest.fixture
def sim() -> LogisticsSimulator:
    """
    Creates a LogisticsSimulator object for testing with two workers and three orders

    Returns:
        A LogisticsSimulator object

    """
    workers = [
        WorkerConfig(
            id="w_1",
            initial_location=Location(lat=0.5, lon=0.5),
            travel_type=WorkerTravelType.WALK,
            speed=0.1,
        ),
        WorkerConfig(
            id="w_2",
            initial_location=Location(lat=0.5, lon=0.5),
            travel_type=WorkerTravelType.WALK,
            speed=0.1,
        ),
    ]

    orders = [
        OrderConfig(
            id="o_1",
            client_id="c_1",
            from_location=Location(lat=0.1, lon=0.1),
            to_location=Location(lat=0.3, lon=0.1),
            creation_time=0,
            time_window=[(0, 0)],
        ),
        OrderConfig(
            id="o_2",
            client_id="c_2",
            from_location=Location(lat=0.4, lon=0.4),
            to_location=Location(lat=0.2, lon=0.3),
            creation_time=10,
            time_window=[(10, 15)],
        ),
        OrderConfig(
            id="o_3",
            client_id="c_2",
            from_location=Location(lat=0.4, lon=0.4),
            to_location=Location(lat=0.2, lon=0.3),
            creation_time=20,
            time_window=[(20, 30)],
        ),
    ]

    order_generator_config = OrderGeneratorConfig(
        generator_type="PredefinedOrderGenerator",
        config=PredefinedOrderGeneratorConfig(orders=orders),
    )

    config = LogisticsSimulatorConfig(
        location_mode=LocationMode.CARTESIAN,
        workers=workers,
        order_generator=order_generator_config,
        start_time=0,
        end_time=40,
        step_size=1,
        order_pickup_time=3,
        order_drop_off_time=5,
        render=RenderConfig(
            render_mode=RenderMode.NONE,
            config=None,
        ),
        seed=42,
    )
    return LogisticsSimulator(config)


def test_reset(sim: LogisticsSimulator) -> None:
    obs, info = sim.reset()

    assert isinstance(obs, Observation)
    assert isinstance(info, Info)

    worker_ids = {"w_1", "w_2"}
    assert len(obs.workers) == 2
    for worker_observation in obs.workers:
        assert worker_observation.id in worker_ids

    order_ids = {"o_1"}
    assert len(obs.orders) == 1
    for order_observation in obs.orders:
        assert order_observation.id in order_ids


def get_noop_action(observation: Observation) -> Action:
    worker_actions = []
    for worker_observation in observation.workers:
        worker_actions.append(
            SpecificWorkerAction(
                worker_id=worker_observation.id,
                action=WorkerAction(type=ActionType.NOOP, parameters=None),
            )
        )
    return Action(worker_actions=worker_actions)


def test_noop_action(sim: LogisticsSimulator) -> None:
    obs, info = sim.reset()
    last_obs = obs
    done = False

    last_reward = None
    while not done:
        action = get_noop_action(obs)
        obs, reward, done, truncated, info = sim.step(action)
        last_obs = obs
        last_reward = reward

        assert isinstance(obs, Observation)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, Info)

        if not done:
            assert reward == 0.0

        for worker_observation in obs.workers:
            assert worker_observation.location == Location(lat=0.5, lon=0.5)
            assert worker_observation.status == WorkerStatus.IDLE

    sim.close()

    assert last_obs.current_time == 40
    assert len(last_obs.orders) == 3
    assert len(last_obs.workers) == 2

    assert last_reward == 3 * -1e6


def get_move_action(observation: Observation, target_location: Location) -> Action:
    worker_actions = []
    for worker_observation in observation.workers:
        action = WorkerAction(type=ActionType.NOOP, parameters=None)
        if worker_observation.status == WorkerStatus.IDLE:
            action = WorkerAction(
                type=ActionType.MOVE,
                parameters=MoveActionParameters(target_location),
            )

        worker_actions.append(SpecificWorkerAction(worker_id=worker_observation.id, action=action))
    return Action(worker_actions=worker_actions)


def test_move_action(sim: LogisticsSimulator) -> None:
    obs, info = sim.reset()
    last_obs = obs
    done = False

    target_location = Location(lat=0.1, lon=0.9)

    while not done:
        action = get_move_action(obs, target_location)
        obs, reward, done, truncated, info = sim.step(action)
        last_obs = obs
    sim.close()

    for worker_observation in last_obs.workers:
        assert target_location.near(worker_observation.location)


def get_fifo_deliver_action(observation: Observation, assigned_orders: set) -> Action:
    worker_actions = []
    for worker_observation in observation.workers:
        action = WorkerAction(type=ActionType.NOOP, parameters=None)
        if worker_observation.status == WorkerStatus.IDLE:
            for order_observation in observation.orders:
                if (
                    order_observation.status == OrderStatus.CREATED
                    and order_observation.id not in assigned_orders
                ):
                    action = WorkerAction(
                        type=ActionType.DELIVER,
                        parameters=DeliverActionParameters(order_id=order_observation.id),
                    )
                    assigned_orders.add(order_observation.id)
                    break

        worker_actions.append(SpecificWorkerAction(worker_id=worker_observation.id, action=action))
    return Action(worker_actions=worker_actions)


def test_deliver_action(sim: LogisticsSimulator) -> None:
    obs, info = sim.reset()
    last_obs = obs
    done = False

    assigned_orders = set()
    total_reward = 0.0
    while not done:
        action = get_fifo_deliver_action(obs, assigned_orders)
        obs, reward, done, truncated, info = sim.step(action)
        last_obs = obs
        total_reward += reward
    sim.close()

    for order_observation in last_obs.orders:
        assert order_observation.status == OrderStatus.COMPLETED

    assert total_reward > -1e6

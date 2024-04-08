import random

import gymnasium as gym
import logistics_envs


def get_idle_action(observation: dict) -> dict:
    n_couriers = len(observation["couriers"]["status"])
    action = {
        "action": [0] * n_couriers,
        "target": [0] * n_couriers,
        "location": [0.0, 0.0] * n_couriers,
    }
    return action


def get_random_move_action(observation: dict) -> dict:
    n_couriers = len(observation["couriers"]["status"])
    action = {
        "action": [0] * n_couriers,
        "target": [0] * n_couriers,
        "location": [0.0, 0.0] * n_couriers,
    }

    couriers = observation["couriers"]
    for i in range(n_couriers):
        if couriers["status"][i] == 0:
            action["action"][i] = 1
            action["location"][i] = [random.random(), random.random()]

    return action


def get_fifo_deliver_action(observation: dict, info: dict, assigned_orders: set) -> dict:
    n_couriers = len(observation["couriers"]["status"])
    action = {
        "action": [0] * n_couriers,
        "target": [0] * n_couriers,
        "location": [0.0, 0.0] * n_couriers,
    }

    couriers = observation["couriers"]
    orders = observation["orders"]
    n_orders = observation["orders"]["n_orders"]
    for courier_index in range(n_couriers):
        if couriers["status"][courier_index] == 0:
            for order_index in range(n_orders):
                order_id = info["order_index_to_id"][order_index]
                if orders["status"][order_index] == 0 and order_id not in assigned_orders:
                    action["action"][courier_index] = 2
                    action["target"][courier_index] = order_index
                    assigned_orders.add(order_id)
                    break

    return action


def main():
    pass
    env = gym.make("logistics_envs/QCommerce-v0", render_mode="human")
    # env = gym.vector.make("logistics_envs/QCommerce-v0", num_envs=3, asynchronous=False)

    observation, info = env.reset()
    done = False

    assigned_orders = set()
    while not done:
        # action = get_idle_action(obs)
        # action = get_random_move_action(observation)
        action = get_fifo_deliver_action(observation, info, assigned_orders)
        observation, reward, done, truncated, info = env.step(action)
        print(reward)

    env.close()


if __name__ == "__main__":
    main()

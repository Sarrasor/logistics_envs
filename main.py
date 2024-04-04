import gymnasium as gym
import logistics_envs


def main():
    pass
    env = gym.make("logistics_envs/VRP-v0")
    # env = gym.vector.make("logistics_envs/VRP-v0", num_envs=3, asynchronous=False)
    obs, info = env.reset()
    done = False
    print(f"Initial Observation: {obs} Info: {info}")

    while not done:
        action = env.action_space.sample()
        observation, reward, done, truncated, info = env.step(action)
        print(
            f"Action: {action} Observation: {observation} Reward: {reward} Done: {done} Truncated: {truncated} Info: {info}"
        )

    env.close()


if __name__ == "__main__":
    main()

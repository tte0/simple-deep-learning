import gymnasium as gym
import numpy as np

def run(epochs):
    rewards_per_epoch = np.zeros(epochs)

    for i in range(epochs):
        env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True, render_mode="human")

        state = env.reset()
        terminated = False
        truncated = False
        while not terminated and not truncated:
            action = env.action_space.sample()
            new_state, reward, terminated, truncated, _ = env.step(action)
            state = new_state
            rewards_per_epoch[i] = reward
        env.close()


if __name__ == "__main__":
    run()
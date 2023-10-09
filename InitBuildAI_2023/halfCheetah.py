import gymnasium as gym
import numpy as np

alpha = 0.1
gamma = 0.9 #discount factor
epsilon=0.1

env = gym.make("HalfCheetah-v4", render_mode="human")
# Reset the environment to its initial state
observation = env.reset()

# Define the number of time steps you want to run the environment
num_steps = 1000

for step in range(num_steps):
    # Render the environment (optional, but can be useful for visualization)
    env.render()

    # Replace this with your control logic
    # In this example, we take random actions
    action = env.action_space.sample()

    # Take a step in the environment
    next_state, reward, done, info, _ = env.step(action)

    # Check if the episode is done (i.e., the simulation terminated)
    if done:
        print("Episode finished after {} timesteps".format(step+1))
        break

# Close the environment (always a good practice)
env.close()
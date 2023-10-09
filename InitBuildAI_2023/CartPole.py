import gym

# Create the CartPole environment
env = gym.make("CartPole-v1", render_mode="human")

# Set the number of episodes and maximum timesteps per episode
num_episodes = 100
max_timesteps = 200

for episode in range(num_episodes):
    # Reset the environment for a new episode
    state = env.reset()
    total_reward = 0

    for t in range(max_timesteps):
        # Render the environment (optional, but helps visualize the simulation)
        env.render()

        # Choose a random action (0: left, 1: right)
        action = env.action_space.sample()

        observation = env.step(action)
        next_state, reward, terminated, truncated, _ = observation


        # Update the total reward for this episode
        total_reward += reward

        if terminated or truncated:
            # if episode % 100 == 0:
            print(f"Episode {episode + 1} finished after {t + 1} timesteps. Total reward: {total_reward}")
            break

# Close the environment
env.close()

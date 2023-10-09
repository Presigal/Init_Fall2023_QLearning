import gym
import numpy as np

alpha = 0.1
gamma = 0.9 #discount factor
epsilon=0.1

env = gym.make("FrozenLake-v1", render_mode="human")
n_states = env.observation_space.n
n_actions = env.action_space.n
Q = np.zeros((n_states, n_actions))

def choose_action(state):
    if np.random.uniform() < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state])
        print(f"State: {state}, Chosen Action: {action}")
    return action

def train(episodes):
    for i in range(episodes):
        state=env.reset()
        done=False

        while not done:
            action=choose_action(state)

            next_state, reward, done, info = env.step(action)

            Q[state][action] = Q[state][action] + alpha*(reward+gamma*np.max(Q[next_state])-Q[state])

            state = next_state

        if (i+1) % 100 == 0:
            print(f"Episode {i+1}: Q-table sum = {np.sum(Q)}")

    print("Final Q-table:")
    print(Q)

train(1000)
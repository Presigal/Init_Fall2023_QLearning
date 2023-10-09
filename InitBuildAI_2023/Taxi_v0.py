Pythoimport gym
import numpy as np

alpha = 0.1
gamma = 0.9 #discount factor
epsilon=0.1

env = gym.make("Taxi-v3", render_mode="human")


n_states = env.observation_space.n
n_actions = env.action_space.n
Q = np.zeros((n_states, n_actions))

num_episodes = 10000

#need to create the random number generator for exploit explore
rng = np.random.default_rng()

def choose_action(state):
    #This if statement is for exploring
    if np.random.uniform() < epsilon:
        action = env.action_space.sample()
    #This else statement is for exploitation
    else:
        action = np.argmax(Q[state])
        print(f"State: {state}, Chosen Action: {action}")
    return action

for i in range(num_episodes):
    #reset the enviroment for the next state
    state, __ = env.reset()
    print(f"*****************State: {state}**********************")
    print(f"*****************State DT: {type(state)}**********************")
    done = False

    action = choose_action(state)
    print(f"*****************State: {state}**********************")
    print(f"*****************Action: {action}**********************")
    print(f"*****************State DT: {type(state)}**********************")
    print(f"*****************Action DT: {type(action)}**********************")
    
    #while the episode is not done
    while not done:
        next_state, reward, done, info, _ = env.step(action)
        
        
        
        state = int(state)
        action = int(action)
    
        Q[state][action] = Q[state][action] + alpha*(reward+gamma*np.max(Q[next_state])-Q[state])
    
        state = next_state
    
        if (i+1) % 100 == 0:
            print(f"Episode {i+1}: Q-table sum = {np.sum(Q)}")
    
print("Final Q-table:")
print(Q)

    

'''



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
            

            #An observation is returned as an int() that encodes the corresponding state, calculated by 
            #((taxi_row * 5 + taxi_col) * 5 + passenger_location) * 4 + destination

            next_state, reward, done, info = env.step(action)

            Q[state][action] = Q[state][action] + alpha*(reward+gamma*np.max(Q[next_state])-Q[state])

            state = next_state

        if (i+1) % 100 == 0:
            print(f"Episode {i+1}: Q-table sum = {np.sum(Q)}")

    print("Final Q-table:")
    print(Q)

train(1000)
gg
'''
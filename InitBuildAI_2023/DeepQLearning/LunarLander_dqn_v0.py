import gym
from dql_v0 import Agent
#from utils import plot_learning_curve
import numpy as np

#test_zero = 100000 % 0
#print(f"If you see this, its a valid value: {test_zero}")

if __name__ == '__main__':
    env = gym.make('LunarLander-v2', render_mode='human')
    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=4,
                  eps_end=0.01, input_dims=[8], lr=0.003)
    scores, eps_history = [], []
    n_games = 500

    for i in range(n_games):
        score=0
        done=False
        observation,_=env.reset()
        while not done:
            action=agent.choose_action(observation)
            observation_, reward, done, info, _ = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward,
                                   observation_, done)
            agent.learn()
            observation = observation_
        scores.append(score)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores[-100:])

        print('episode', i, 'score %.2f' %score,
              'average score %.2f' % avg_score,
              'epsilon %.2f' % agent.epsilon)
        

        '''
        x = [i+1 for i in range(n_games)]
        filename = 'lunar_lander_2020.png'
        plot_learning_curve(x, scores, eps_history, filename)
        #plot learning
        '''


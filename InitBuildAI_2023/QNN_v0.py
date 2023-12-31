import numpy as np
import tensorflow as tf
from tensorflow import keras

class ReplayBuffer():
    def __init__(self, mem_size, input_dims):
        self.mem_size = mem_size
        self.mem_cntr = 0

        self.state_memory = np.zeros(self.mem_size *(input_dims),
            dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims),
                                         dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.int32)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index]=reward
        self.action_memory[index]=action
        self.terminal_memory[index]= 1 - int(done)

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states=self.state_memory[batch]
        states_=self.new_state_memory[batch]
        rewards = self.reward_memory[batch]


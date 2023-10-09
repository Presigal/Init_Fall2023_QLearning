import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

env = gym.make("FrozenLake-v1", desc=None, map_name = "8x8", is_slippery=True, render_mode="rgb_array")
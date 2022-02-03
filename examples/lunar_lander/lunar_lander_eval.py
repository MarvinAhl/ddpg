import sys
sys.path.insert(1, '../..')

from ddpg import DDPG

import torch

import gym

import numpy as np
from matplotlib import pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device {device}')

env = gym.make('LunarLanderContinuous-v2')

nS = env.observation_space.shape[0]
nA = env.action_space.shape[0]

agent = DDPG(nS, nA, explore_decay_steps=10000, buffer_size_max=30000,
             buffer_size_min=512, device=device)
agent.load_net('lunar_lander.net')

while True:
    obsv, done = env.reset(), False

    while not done:
        env.render()

        actions = agent.act(obsv)
        new_obsv, _, _, _ = env.step(actions)

        obsv = new_obsv
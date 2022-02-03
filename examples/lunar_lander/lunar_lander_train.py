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

agent = DDPG(nS, nA, policy_hidden=(128, 128), value_hidden=(128, 128), explore_decay_steps=30000, buffer_size_max=30000,
             buffer_size_min=256, device=device)

episode = 0
while True:
    try:
        episode += 1
        print(f'Episode {episode} started')
        obsv, done = env.reset(), False

        while not done:
            env.render()

            actions = agent.act_explore(obsv)

            new_obsv, reward, done, info = env.step(actions)

            time_out = 'TimeLimit.truncated' in info and info['TimeLimit.truncated']
            terminal = done and not time_out

            agent.experience(obsv, actions, reward, new_obsv, terminal)
            agent.train()

            obsv = new_obsv
    except KeyboardInterrupt:
        break

agent.save_net('lunar_lander.net')
"""
Author: Marvin Ahlborn

This is an implementation of the Deep Deterministic Policy Gradient (DDPG)
Algorithm using PyTorch.
"""

import torch
from torch import tensor
from torch import nn
import numpy as np

class PolicyNetwork(nn.Module):
    """
    The deterministic Policy Network
    """
    def __init__(self, state, actions, hidden=(512, 512)):
        """
        state: Integer telling the state dimension
        actions: Integer telling the number of actions
        hidden: Tuple of the each hidden layers nodes
        """
        super(PolicyNetwork, self).__init__()

        modules = []

        modules.append(nn.Linear(state, hidden[0]))
        modules.append(nn.LeakyReLU(0.1))

        for i in range(len(hidden) - 1):
            modules.append(nn.Linear(hidden[i], hidden[i+1]))
            modules.append(nn.LeakyReLU(0.1))
        
        modules.append(nn.Linear(hidden[-1], actions))
        modules.append(nn.Tanh())  # Action Output is float between -1 and 1

        self.module_stack = nn.Sequential(*modules)
    
    def forward(self, state):
        actions = self.module_stack(state)
        return actions

class ValueNetwork(nn.Module):
    """
    Q-Network
    """
    def __init__(self, state, actions, hidden=(512, 512)):
        """
        state: Integer telling the state dimension
        actions: Integer telling the number of actions
        hidden: Tuple of the each hidden layers nodes
        """
        super(ValueNetwork, self).__init__()

        modules = []

        modules.append(nn.Linear(state + actions, hidden[0]))  # Takes in State and Action
        modules.append(nn.LeakyReLU(0.1))

        for i in range(len(hidden) - 1):
            modules.append(nn.Linear(hidden[i], hidden[i+1]))
            modules.append(nn.LeakyReLU(0.1))
        
        modules.append(nn.Linear(hidden[-1], 1))  # Only one output for the State-Action-Value

        self.module_stack = nn.Sequential(*modules)
    
    def forward(self, state, actions):
        state_action_input = torch.cat((state, actions), dim=-1)
        value_output = self.module_stack(state_action_input)
        return value_output

class ReplayBuffer:
    """
    Prioritizing Replay Buffer
    """
    def __init__(self, state, actions, max_len=50000, alpha=0.6, beta=0.1, beta_increase_steps=20000):
        """
        state: Dimension of State
        actions: Number of Actions
        """
        self.states = np.empty((max_len, state), dtype=np.float32)
        self.actions = np.empty((max_len, actions), dtype=np.int64)
        self.rewards = np.empty(max_len, dtype=np.float32)
        self.next_states = np.empty((max_len, state), dtype=np.float32)
        self.terminals = np.empty(max_len, dtype=np.int8)
        self.errors = np.empty(max_len, dtype=np.float32)  # The Q-Network temporal difference error used for training
        self.weights = np.empty(max_len, dtype=np.float32)  # Weights for gradient descend bias correction
        self.probabilities = np.empty(max_len, dtype=np.float32)

        self.index = 0
        self.full = False
        self.max_len = max_len
        self.rng = np.random.default_rng()
        self.probs_updated = False  # Indicates whether probabilities have to be recalculated
        self.alpha = alpha  # Blend between uniform distribution (alpha = 0) and Probabilities according to rank (alpha = 1)
        self.beta = beta  # Blend between full bias correction (beta = 1) and no bias correction (beta = 0)
        self.beta_increase = (1.0 - beta) / beta_increase_steps  # Adder to linearly reach 1 within given amount of steps
    
    def store_experience(self, state, actions, reward, next_state, terminal):
        """
        Stores given SARS Experience in the Replay Buffer.
        Returns True if the last element has been written into memory and
        it will start over replacing the first elements at the next call.
        """
        self.states[self.index] = state
        self.actions[self.index] = actions
        self.rewards[self.index] = reward
        self.next_states[self.index] = next_state
        self.terminals[self.index] = terminal
        self.errors[self.index] = 1.0 if self.__len__() == 0 else self.errors[:self.__len__()].max()  # To make it likely picked the first time

        self.probs_updated = False
        
        self.index += 1
        self.index %= self.max_len  # Replace oldest Experiences if Buffer is full

        if self.index == 0:
            self.full = True
            return True
        return False
    
    def update_experiences(self, indices, errors):
        """
        Update TD Errors for elements given by indices. Should be called after they have
        been replayed and new errors were calculated. Errors have to be input as absolute values.
        """
        self.errors[indices] = errors
        self.probs_updated = False

    def get_experiences(self, batch_size):
        """
        Returns batch of experiences for replay.
        """
        buff_len = self.__len__()
        
        if not self.probs_updated:
            sorted_indices = self.errors[:buff_len].argsort()[::-1]  # Indices from highest to lowest error
            ranks = np.arange(buff_len)[sorted_indices] + 1
            scaled_priorities = (1 / ranks)**self.alpha
            
            self.probabilities[:buff_len] = scaled_priorities / scaled_priorities.sum()
            unnormed_weights = (self.probabilities[:buff_len] * buff_len)**-self.beta
            self.weights[:buff_len] = unnormed_weights / unnormed_weights.max()

            self.beta += self.beta_increase  # Update beta
            self.beta = min(1.0, self.beta)

            self.probs_updated = True

        indices = self.rng.choice(np.arange(buff_len), batch_size, p=self.probabilities[:buff_len])

        weights = self.weights[indices]
        states = self.states[indices]
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        next_states = self.next_states[indices]
        terminals = self.terminals[indices]

        return indices, weights, states, actions, rewards, next_states, terminals

    def __len__(self):
        return self.max_len if self.full else self.index

class BDQ:
    def __init__(self, state, actions, shared=(512, 512), branch=(128, 128), gamma=0.99, learning_rate=0.0005,
                 weight_decay=0.0001, epsilon_start=1.0, epsilon_decay_steps=20000,
                 epsilon_min=0.1, new_actions_prob=0.05, buffer_size_max=50000, buffer_size_min=1000,
                 batch_size=64, replays=1, tau=0.01, alpha=0.6, beta=0.1, beta_increase_steps=50000, device='cpu'):
        """
        state: Integer of State Dimension
        actions: Tuple of Subactions per Action
        """
        self.state = state
        self.actions = actions
        self.shared = shared
        self.branch = branch
        self.q_net = Network(state, actions, shared, branch).to(device)
        self.target_q_net = Network(state, actions, shared, branch).to(device)
        self._update_target(1.0)  # Fully copy Online Net weights to Target Net

        self.optimizer = torch.optim.RMSprop(self.q_net.parameters(),  # Using RMSProp because it's more stable, not as aggressive than Adam
                                             lr=learning_rate, weight_decay=weight_decay)
        self.weight_decay = weight_decay

        self.buffer = ReplayBuffer(state, len(actions), buffer_size_max, alpha, beta, beta_increase_steps)
        self.buffer_size_max = buffer_size_max
        self.buffer_size_min = buffer_size_min
        self.batch_size = batch_size
        self.replays = replays  # On how many batches it should train after each step
        self.alpha = alpha
        self.beta = beta
        self.beta_increase_steps = beta_increase_steps

        # Can be calculated by exp(- dt / lookahead_horizon)
        self.gamma = gamma  # Reward discount rate

        self.learning_rate = learning_rate

        self.rng = np.random.default_rng()

        # Linearly decay Epsilon from start to min in a given amount of steps
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_decay = (epsilon_start - epsilon_min) / epsilon_decay_steps
        self.epsilon_min = epsilon_min

        # How many times per second on average a new action is chosen randomly (1 is regular e-greedy)
        # Can be calculated by this formula: new_actions_per_second * dt
        self.new_actions_prob = new_actions_prob
        self.rand_actions = self.rng.integers(actions, dtype=np.int64)

        self.tau = tau  # Mixing parameter for polyak averaging of target and online network

        self.device = device
    
    def reset(self):
        """
        Reset object to its initial state if you want to do multiple training passes with it
        """
        self.q_net = Network(self.state, self.actions, self.shared, self.branch).to(self.device)
        self.target_q_net = Network(self.state, self.actions, self.shared, self.branch).to(self.device)
        self._update_target(1.0)  # Fully copy Online Net weights to Target Net

        self.optimizer = torch.optim.RMSprop(self.q_net.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        self.buffer = ReplayBuffer(self.state, len(self.actions), self.buffer_size_max, self.alpha, self.beta, self.beta_increase_steps)

        self.epsilon = self.epsilon_start

        self.rng = np.random.default_rng()
    
    def act(self, state):
        """
        Decides on action based on current state using epsilon-greedy Policy.
        Choses if action is random for each action individually.
        Random actions are correlated though. Every time there is a chance that
        the random actions from last time are used again.
        """
        actions = self.act_optimally(state)  # Greedy actions

        is_rand = self.rng.random(len(self.actions)) < self.epsilon  # List of Booleans indicating which actions will be random

        probs_new = self.new_actions_prob / self.epsilon  # Probability for new random action
        is_new = self.rng.random(len(self.actions)) < probs_new  # List of Booleans indicating which random actions will be exchanged by new ones

        self.rand_actions[is_rand * is_new] = self.rng.integers(self.actions, dtype=np.int64)[is_rand * is_new]  # Generate new random actions

        actions[is_rand] = self.rand_actions[is_rand]  # Make some actions random

        self._update_parameters()

        return actions

    def act_optimally(self, state):
        """
        Decides on action based on current state using greedy Policy.
        """
        state = tensor(state, device=self.device, dtype=torch.float32).unsqueeze(0)
        qs = self.q_net(state)

        actions = np.empty(len(self.actions), dtype=np.int64)
        for i, q in enumerate(qs):
            actions[i] = q.detach().unsqueeze(0).argmax().item()

        return actions
    
    def experience(self, state, action, reward, next_state, terminal):
        """
        Takes experience and stores it for replay.
        """
        if self.buffer.store_experience(state, action, reward, next_state, terminal):
            # Reset exploration rate when Replay Buffer is full to always have
            # negative Experiences in storage. Will prevent catastrophic forgetting
            self.epsilon = self.epsilon_start  
    
    def train(self):
        """
        Train Q-Network on a batch from the replay buffer.
        """
        if len(self.buffer) < self.buffer_size_min:
            return  # Dont train until Replay Buffer has collected a certain number of initial experiences

        for _ in range(self.replays):
            indices, weights, states, actions, rewards, next_states, terminals = self.buffer.get_experiences(self.batch_size)

            weights = torch.from_numpy(weights).to(self.device)
            states = torch.from_numpy(states).to(self.device)
            rewards = torch.from_numpy(rewards).to(self.device)
            actions = torch.from_numpy(actions).to(self.device)
            next_states = torch.from_numpy(next_states).to(self.device)
            terminals = torch.from_numpy(terminals).to(self.device)

            qs = self.q_net(next_states)
            target_qs = self.target_q_net(next_states)

            target_values = torch.empty((self.batch_size, len(self.actions)), device=self.device, dtype=torch.float32)
            for i, (q, target_q) in enumerate(zip(qs, target_qs)):
                max_actions = q.detach().argmax(-1).unsqueeze(-1)
                max_action_qs = target_q.detach().gather(-1, max_actions).squeeze()
                target_values[:, i] = max_action_qs

            mean_target_values = target_values.mean(-1)  # One Target for each 
            targets = rewards + self.gamma * mean_target_values * (1 - terminals)

            qs = self.q_net(states)
            predictions = torch.empty((self.batch_size, len(self.actions)), device=self.device, dtype=torch.float32)
            for i, q in enumerate(qs):
                prediction = q.gather(-1, actions[:, i].unsqueeze(1)).squeeze()
                predictions[:, i] = prediction

            td_errors = targets.unsqueeze(1).expand_as(predictions) - predictions
            weighted_td_errors = weights.unsqueeze(1).expand_as(td_errors) * td_errors  # PER Bias correction
            loss = weighted_td_errors.pow(2).mean()  # Square Loss like in Paper

            self.optimizer.zero_grad()
            loss.backward()

            # Gradient Rescaling of Shared Network part because all branches propagate gradients back through it
            for i, param in enumerate(self.q_net.parameters()):
                if i < len(self.shared) * 2:
                    param.grad /= len(actions) + 1

            self.optimizer.step()

            errors = td_errors.detach().abs().sum(-1).cpu().numpy()  # Sum of absolute TD Errors used for Replay Prioritization
            self.buffer.update_experiences(indices, errors)  # Update replay buffer's temporal difference errors

        self._update_target(self.tau)
    
    def save_net(self, path):
        torch.save(self.q_net.state_dict(), path)
    
    def load_net(self, path):
        self.q_net.load_state_dict(torch.load(path))
        self._update_target(1.0)  # Also load weights into target net
    
    def _update_target(self, tau):
        """
        Update Target Network by blending Target und Online Network weights using the factor tau (Polyak Averaging)
        A tau of 1 just copies the whole online network over to the target network
        """
        for param, target_param in zip(self.q_net.parameters(), self.target_q_net.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
    
    def _update_parameters(self):
        """
        Decay epsilon
        """
        self.epsilon -= self.epsilon_decay
        self.epsilon = max(self.epsilon, self.epsilon_min)
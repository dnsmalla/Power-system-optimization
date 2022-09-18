import numpy as np
import copy
from collections import deque
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

# GPU
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
# print(device)  # cuda:0
torch.manual_seed(100)

# 1.Buffer


class ReplayBuffer:
    def __init__(self, memory_size=int(1e+6)):
        self.memory = deque([], maxlen=memory_size)
        self.is_gpu = torch.cuda.is_available
        return

    def cache(self, state, next_state, action, reward, done):
        if self.is_gpu:
            state = torch.tensor(state, dtype=torch.float).cpu()  # .cuda() -> .cpu() masa)
            next_state = torch.tensor(next_state, dtype=torch.float).cpu()  # .cuda() -> .cpu() masa
            action = torch.tensor(action, dtype=torch.float).cpu()  # .cuda() -> .cpu() masa
            reward = torch.tensor(reward).cpu()  # .cuda() -> .cpu() masa
            done = torch.tensor([done]).cpu()  # .cuda() -> .cpu() masa
        else:
            state = torch.tensor(state, dtype=torch.float)
            next_state = torch.tensor(next_state, dtype=torch.float)
            action = torch.tensor(action, dtype=torch.float)
            reward = torch.tensor(reward)
            done = torch.tensor([done])
        self.memory.append((state, next_state, action, reward, done))

    def sample(self, batch_size=64):
        batch = random.sample(self.memory, 64)  # adding a random comma somehow solved the program?
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action, reward.squeeze(), done.squeeze()


# 2.Model
# Actor network
class PolicyNetwork(nn.Module):
    def __init__(self, num_state, num_action, hidden_size=64):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(num_state, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_action)

    def forward(self, x, index):
        x = x[:, index]
        h = F.relu(self.fc1(x))
        h = F.sigmoid(self.fc2(h))
        action = self.fc3(h)
        return action


# Critic network
class QNetwork(nn.Module):
    def __init__(self, num_state, num_action, agent_num, hidden_size=64, init_w=3e-3):
        super(QNetwork, self).__init__()
        input_size1 = num_state * agent_num
        input_size2 = hidden_size + num_action * agent_num
        self.fc1 = nn.Linear(input_size1, hidden_size)
        self.fc2 = nn.Linear(input_size2, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)

    def forward(self, states, actions):
        states = states.view(states.size()[0], -1)
        actions = actions.view(actions.size()[0], -1)
        h = F.relu(self.fc1(states))
        x = torch.cat([h, actions], 1)
        h = F.sigmoid(self.fc2(x))
        q = self.fc3(h)
        return q


# Ornstein–Uhlenbeck process noise for action using
class OrnsteinUhlenbeckProcess:
    def __init__(self, theta=0.15, mu=0.0, sigma=0.2, dt=1e-2, x0=None, size=1, sigma_min=None, n_steps_annealing=1000):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.size = size
        self.num_steps = 0

        self.x_prev = self.x0 if self.x0 is not None else np.zeros(self.size)

        if sigma_min is not None:
            self.m = -float(sigma - sigma_min) / float(n_steps_annealing)
            self.c = sigma
            self.sigma_min = sigma_min
        else:
            self.m = 0
            self.c = sigma
            self.sigma_min = sigma

    def current_sigma(self):
        sigma = max(self.sigma_min, self.m * float(self.num_steps) + self.c)
        return sigma

    def sample(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.current_sigma() * np.sqrt(
            self.dt) * np.random.normal(size=self.size)
        self.x_prev = x
        self.num_steps += 1
        return x


# Maddpg Agent inputs (state, action)
class MaddpgAgents:
    def __init__(self, observation_space, action_space, num_agent, gamma=0.98, lr=0.001, batch_size=1024,
                 memory_size=int(1e6), tau=0.1, grad_norm_clipping=1):
        self.num_state = observation_space
        self.num_action = action_space
        self.n = num_agent
        self.gamma = gamma
        self.actor_group = [PolicyNetwork(self.num_state, self.num_action).to(device) for _ in
                            range(self.n)]
        self.target_actor_group = copy.deepcopy(self.actor_group)
        self.actor_optimizer_group = [optim.Adam(self.actor_group[i].parameters(), lr=0.001
                                                 ) for i in range(self.n)]
        self.critic_group = [QNetwork(self.num_state, self.num_action, self.n).to(device) for _ in range(self.n)]
        self.target_critic_group = copy.deepcopy(self.critic_group)
        self.critic_optimizer_group = [optim.Adam(self.critic_group[i].parameters(), lr=lr) for i in range(self.n)]
        self.buffer = ReplayBuffer(memory_size=memory_size)
        self.loss_fn = torch.nn.MSELoss()
        self.batch_size = batch_size
        self.is_gpu = torch.cuda.is_available
        self.noise = OrnsteinUhlenbeckProcess(size=self.num_action)
        self.grad_norm_clipping = grad_norm_clipping
        self.tau = tau

    @torch.no_grad()
    def td_targeti(self, reward, state, next_state, done, agent_index):
        next_actions = []
        for i in range(self.n):
            actionsi = torch.tanh(self.target_actor_group[i](state, i))
            actionsi = actionsi[:, np.newaxis, :]
            next_actions.append(actionsi)
        next_actions = torch.cat(next_actions, dim=1)
        next_q = self.target_critic_group[agent_index](next_state, next_actions)
        if self.n != 1:
            reward = reward[:, agent_index]
            done = done[:, agent_index]
        reward = reward[:, np.newaxis]
        done = done[:, np.newaxis]
        done = torch.tensor(done, dtype=torch.int)
        td_targeti = reward + self.gamma * next_q * (1. - done.data)
        return td_targeti.float()

    def update(self):
        for i in range(self.n):
            state, next_state, action, reward, done = self.buffer.sample(self.batch_size)
            td_targeti = self.td_targeti(reward, state, next_state, done, i)
            current_q = self.critic_group[i](state, action)

            # critic update
            critic_loss = self.loss_fn(current_q, td_targeti)
            self.critic_optimizer_group[i].zero_grad()
            critic_loss.backward()
            clip_grad_norm_(self.critic_group[i].parameters(), max_norm=self.grad_norm_clipping)
            self.critic_optimizer_group[i].step()

            # actor update
            ac = action.clone()
            ac_up = self.actor_group[i](state, i)
            ac[:, i, :] = torch.tanh(ac_up)
            pr = -self.critic_group[i](state, ac).mean()
            pg = (ac[:, i, :].pow(2)).mean()
            actor_loss = pr + pg * 1e-3
            self.actor_optimizer_group[i].zero_grad()
            clip_grad_norm_(self.actor_group[i].parameters(), max_norm=self.grad_norm_clipping)
            actor_loss.backward()
            self.actor_optimizer_group[i].step()

        # soft-update
        for i in range(self.n):
            for target_param, local_param in zip(self.target_actor_group[i].parameters(),
                                                 self.actor_group[i].parameters()):
                target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
            for target_param, local_param in zip(self.target_critic_group[i].parameters(),
                                                 self.critic_group[i].parameters()):
                target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    # def get_action(self,state,greedy=False):
    def get_action(self, state, greedy=False):
        state = torch.tensor(state, dtype=torch.float).cpu()  # .cuda() -> .cpu() masa
        state = state[np.newaxis, :, :]
        actions = []
        for i in range(self.n):
            action = torch.tanh(self.actor_group[i](state, index=i))
            if not greedy:
                action += torch.tensor(self.noise.sample(), dtype=torch.float).cpu()
            actions.append(action)
        actions = torch.cat(actions, dim=0)
        return np.clip(actions.detach().cpu().numpy(), -1.0, 1.0)

import random
from maddpg import MaddpgAgents
from environments import Environment
import re
import tqdm
import numpy as np
import argparse
import os
import pandas as pd
from matplotlib import pyplot
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
epsilon = 0.3
max_ep = 0
all_episodes = []
# ------args-------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Set parameters.')
parser.add_argument('--num_episode', default=1000, type=int, help='Number of episode')
parser.add_argument('--max_steps', default=24, type=int, help='Number of max steps')
parser.add_argument('--memory_size', default=10000, type=int, help='Number of memory size')
parser.add_argument('--initial_memory_size', default=100000, type=int, help='Number of initial memory size')
parser.add_argument('--lr', default=0.01, type=float, help='Learning rate of Agents')

# agent config
parser.add_argument('--num_pv', default=3, type=int, help='Number of PV Agents')
parser.add_argument('--observation_space', default=3, type=int, help='Number of PV Agents')
# demand config
parser.add_argument('--demand', default="constant", type=str, help='type of Demand')

args = parser.parse_args()

episode_rewards = []
num_average_epidodes = 100

env = Environment(args)
agent = MaddpgAgents(observation_space=args.observation_space, action_space=1, num_agent=args.num_pv, lr=args.lr,
                     memory_size=args.memory_size)

# -------- plot用 ----------
reward_log = []
episode_reward_log = []

config_nums = []
agent1_log = []
agent2_log = []
agent3_log = []
max_sc_1 = []
max_sc_2 = []
max_sc_3 = []
steps = 0
counter = 0
for episode in tqdm.trange(args.num_episode):
    a = 0
    obs_n = env.reset()
    next_obs_n = obs_n
    episode_reward = 0
    reward_sum = [0]
    all_episodes.append(episode)
    agent1_episode_log = []
    agent2_episode_log = []
    agent3_episode_log = []
    # -------- plot用' ----------
    # print("env.config_epi", env.config["solar_insolations"][-3:])
    schedule_1 = []
    schedule_2 = []
    schedule_3 = []

    for t in range(24):
        assert len(obs_n) == args.num_pv
        total_reward = []
        env.steps = t
        action_n = agent.get_action(obs_n)
        p = random.random()
        if p < epsilon:
            for i in range(3):
                a = random.randint(0, 1)
                env.operate[i] = a
            schedule_1.append(env.operate[0])
            schedule_2.append(env.operate[1])
            schedule_3.append(env.operate[2])
        else:
            assert len(action_n) == args.num_pv
            next_obs_n, reward_n, done, _ = env.step(action_n, obs_n, counter)
            schedule_1.append(env.operate[0])
            schedule_2.append(env.operate[1])
            schedule_3.append(env.operate[2])
        steps = steps + 1
        if steps % 24 == 0:
            counter += 1
        for i in range(args.num_pv):
            total_reward.append(sum(reward_n))
        episode_reward += sum(total_reward)

        reward_sum = [x + y for (x, y) in zip(reward_sum, total_reward)]  # plot用
        num_steps = t  # plot用
        agent.buffer.cache(obs_n, next_obs_n, action_n, total_reward, done)
        if steps % 24 != 0:
            obs_n = next_obs_n
        else:
            obs_n = env.reset()

        # -------- plot用 ----------
        if action_n[0][0] < 0:
            action_n[0][0] = 0
        else:
            action_n[0][0] = 1
        if action_n[1][0] < 0:
            action_n[1][0] = 0
        else:
            action_n[1][0] = 1
        if action_n[2][0] < 0:
            action_n[2][0] = 0
        else:
            action_n[2][0] = 1
        agent1_episode_log.append(action_n[0][0])
        agent2_episode_log.append(action_n[1][0])
        agent3_episode_log.append(action_n[2][0])

        if t == 23:
            reward_sum = [x / num_steps for x in reward_sum]  # plot用
            reward_log.append(reward_sum)  # plot用
            episode_reward_log.append(episode_reward / num_steps)  # plot用
            agent1_log.append(agent1_episode_log)
            agent2_log.append(agent2_episode_log)
            agent3_log.append(agent3_episode_log)
            break
    np.save('./reward_log/episode_reward_log_{0}'.format("0620"), episode_reward_log)  # plot用

    if episode > 40 and episode % 4 == 0:
        agent.update()
    episode_rewards.append(episode_reward)
    if episode % 200 == 0:
        print("Episode %d finished | Episode reward %f" % (episode, episode_reward_log[episode]))
    if episode > 2:
        print('max', max(episode_reward_log))
        print('rn', episode_reward_log[episode])
        if episode_reward_log[episode] == max(episode_reward_log):
            print('YO!')
            max_ep = episode
            max_sc_1 = schedule_1[:]
            max_sc_2 = schedule_2[:]
            max_sc_3 = schedule_3[:]

col1 = "Episodes, x"
col2 = "Rewards, y"

print(len(all_episodes))
print(len(episode_reward_log))
data = pd.DataFrame({col1: all_episodes, col2: episode_reward_log})
data.to_excel('rewardepisodes.xlsx', sheet_name='sheet1', index=False)


print("args", args)
print('maximum ep was', max_ep)
actions =np.array([max_sc_1,
                max_sc_2,
                max_sc_3])
print(actions)
episode_reward_log = np.load('./reward_log/episode_reward_log_0620.npy', allow_pickle=True)
pyplot.plot(episode_reward_log)
pyplot.show()
pyplot.plot
episode_reward_log2 = pd.DataFrame(episode_reward_log)
t_average = episode_reward_log2.rolling(window=100).mean()
# plt.plot(t_average, 'k-', label='moving ave reward')
# plt.show()
from env_vis import EPlot
EPlot(actions, env)
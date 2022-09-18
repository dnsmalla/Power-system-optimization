import numpy as np
from matplotlib import pyplot
episode_reward_log = np.load('./reward_log/episode_reward_log_0620.npy', allow_pickle=True)

# plt.plot()
episode_reward_log = np.load('./reward_log/episode_reward_log_0620.npy', allow_pickle=True)
pyplot.plot(episode_reward_log)
pyplot.show()

import sys
import numpy as np
import random

random.seed(2022)


class Environment:
    """main setting for multi user """
    total_supply: int

    def __init__(self, args):
        self.finish_oper = None
        self.steps = None
        self.oper_time = None
        self.args = args
        self.n = args.num_pv  # Agent
        self.money_reward = 0
        self.supply = 0
        self.total_consumption = 0
        self.total_supply = 0
        self.operate = [0, 0, 0]
        self.use_pv_list = [0,1]
        self.agents = ["agent1", "agent2", "agent3"]
        self.num_agent = len(self.agents)
        self.config = {"SolarPV": (0, 0, 0, 0, 50, 50,
                                   100, 200, 300, 300, 400, 700,
                                   700, 400, 300, 300, 200, 100,
                                   50, 50, 0, 0, 0, 0),
                       "SolarPV2": (0, 0, 0, 0, 25, 25,
                                    150, 200, 350, 350, 350, 650,
                                    650, 350, 350, 350, 200, 150,
                                    25, 25, 0, 0, 0, 0),
                       "SolarPV3": (0, 0, 10, 10, 25, 25,
                                    110, 110, 220, 220, 800, 800,
                                    400, 320, 320, 150, 90, 80,
                                    25, 25, 10, 10, 0, 0),
                       "SolarPV4": (0, 0, 40, 60, 60, 60,
                                    100, 100, 100, 100, 100, 100,
                                    60, 60, 60, 60, 60, 30,
                                    0, 0, 0, 0, 0, 0),
                       "SolarPV5": (0, 0, 0, 0, 0, 0,
                                    400, 500, 100, 50, 50, 50,
                                    300, 700, 20, 60, 30, 10,
                                    20, 0, 0, 0, 0, 0),
                       "SolarPV6": (0, 0, 0, 0, 0, 0,
                                    200, 300, 350, 300, 200, 300,
                                    450, 500, 50, 100, 150, 50,
                                    20, 0, 0, 0, 0, 0),
                       "Grid_price": 10,
                       "agent1": [600, 5, 19, 19, 0],  # [kWh/h, operation time, operation limit]
                       "agent2": [125, 9, 15, 23, 7],
                       "agent3": [50, 8, 16, 20, 7]}

        """
        self.config = { "SolarPV":[150, 150, 150, 150, 150, 150,
                                    150, 150, 150, 150, 150, 150,
                                    150, 150, 150, 150, 150, 150,
                                    150, 150, 150, 150, 150, 150]}
        """
    @property
    def hour(self):
        """decide hour """
        return self.steps % 24

    @property
    def next_hour(self):
        """decide next hour"""
        return (self.steps + 1) % 24

    def _reset_total_supply(self, balance):
        if 0 <= balance:
            return
        self.total_supply = 0

    def reset(self):
        # state reset (time, num_activate)
        observations = [[0, 1, 0], [0, 1, 0], [0, 1, 0]]
        # reward reset
        self.total_consumption = 0
        self.total_supply = 0
        self.flag = [0, 0, 0]
        self.finish_oper = [1, 1, 1]
        self.oper_time = [self.config["agent1"][1], self.config["agent2"][1], self.config["agent3"][1]]
        return observations

    def step(self, action_n, obs_n, steps):
        pvlist = ["SolarPV", "SolarPV2", "SolarPV3", "SolarPV4", "SolarPV5", "SolarPV6"]

        if self.steps == 0:
            self.agent1_act = []
            self.agent2_act = []
            self.agent3_act = []

        # setting of pv use from config pv set
        if self.use_pv_list == "random":
            use_pv = self.config[random.choice(pvlist)][self.steps]
        elif len(self.use_pv_list) > 1:
            use_pv = sum(np.array(self.config[pvlist[i]]) for i in self.use_pv_list)
        else:
            use_pv = self.config[pvlist[self.use_pv_list[0]]]
        self.pv_gen = use_pv
        self.supply = use_pv[self.steps]

        penalty = [0 for _ in range(self.num_agent)]

        done = [False for _ in range(self.n)]

        self.operate = [0 for _ in range(self.num_agent)]

        # operate calculation from action taken by agents
        for i, act in enumerate(action_n.flatten()):
            if 0 < act:
                self.operate[i] = 1  # ON
                self.flag[i] = 1
            else:
                self.operate[i] = 0  # OFF
            if i == 0:
                self.agent1_act.append(self.operate[0])
            elif i == 1:
                self.agent2_act.append(self.operate[1])
            elif i == 2:
                self.agent3_act.append(self.operate[2])
        # reward calculation
        cost: int = 0
        need_p = 0.6 * self.flag[0] + 0.125 * self.flag[1] + 0.05 * self.flag[2] 
        if (self.supply/1000) > need_p:
            balance = need_p
        elif self.supply==0 and need_p==0:
            balance = 2
        else:
            balance = -10
        #balance = 10*(self.supply/1000)-1*(0.6 * self.flag[0] * self.finish_oper[0] + 0.125 * self.flag[1] * self.finish_oper[1] + 0.05 * self.flag[2] * self.finish_oper[2])
        if 2 < self.agent1_act.count(1)<6  or 6 < self.agent2_act.count(1) < 10 or 5 < self.agent3_act.count(1) < 9:
            balance += 1
        # self._reset_total_supply(balance)
        cost += balance  # ** 2
        rew_n = [cost, cost, cost]

        for i in range(len(obs_n)):
            if obs_n[i][1] < 0 and self.operate[i] == 1:  # no pV time more than 1 user penalty
                rew_n[i] -= 100
        for i, act in enumerate(action_n.flatten()):
            if self.steps < self.config[self.agents[i]][4] and self.operate[i] == 1:  # early morning open penalty
                rew_n[i] -= 100
            elif self.steps > self.config[self.agents[i]][3] and self.operate[i] == 1:  # early morning open penalty
                rew_n[i] -= 100
            elif sum(self.operate) > 2 and act > 0 and self.supply == 0:  # more than 2 at same time penalty
                rew_n[i] -= 100

        for i, act in enumerate(action_n.flatten()):
            if self.oper_time[i] > 0 and self.flag[i] == 1:
                self.oper_time[i] -= 1
                if self.oper_time[i] == 0:
                    self.finish_oper[i] = 0

        if (self.steps >= self.config["agent1"][3] and self.agent1_act.count(1) < self.config["agent1"][1]) or (
                self.steps >= self.config["agent2"][3] and self.agent2_act.count(1) < self.config["agent2"][1]) or (
                self.steps >= self.config["agent3"][3] and self.agent3_act.count(1) < self.config["agent3"][1]):
            done = [True for _ in range(self.n)]

        if self.steps == 0:
            done = [True for _ in range(self.n)]

        # next_obs calculation

        operation_time = [0 for _ in range(self.num_agent)]
        terminal = [1 for _ in range(self.num_agent)]
        acts_list=[self.agent1_act,self.agent2_act,self.agent3_act]
        for i, acts in enumerate(acts_list):
            if (self.config[self.agents[i]][1] - acts.count(1))>0:
                operation_time[i] = 1
                terminal[i] -= self.operate[i]
            else:
                operation_time[i] = self.config[self.agents[i]][1] - acts.count(1)
                terminal[i] = -1

        next_obs_n = [[(self.steps + 1)*terminal[i]/24, operation_time[i]*self.config[self.agents[i]][0]/self.supply, self.supply/1000] for i in range(self.n)]
        # print("next_obs_n,rew_n,done,steps",next_obs_n,rew_n,done,self.steps,action_n.flatten(),agent1)
        return next_obs_n, rew_n, done, {}

    def render(self):
        pass

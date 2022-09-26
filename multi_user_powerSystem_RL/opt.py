import pyomo.environ as pyo
from pyomo.environ import Binary, NonNegativeReals, maximize, minimize
import gurobipy
import numpy as np
import random
from model_vis import MPlot


class MultiUser():

    def __init__(self) -> None:
        self.use_pv_list = [1]
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
                       "agent1": [600, 5, 19, 19, 1],  # [kWh/h, operation time, operation limit]
                       "agent2": [125, 9, 15, 23, 7],
                       "agent3": [50, 8, 16, 20, 7]}
        self.t_len = len(self.config["SolarPV"])

    def pv_use(self):
        pvlist = ["SolarPV", "SolarPV2", "SolarPV3", "SolarPV4", "SolarPV5", "SolarPV6"]
        # setting of pv use from config pv set
        if self.use_pv_list == "random":
            use_pv = self.config[random.choice(pvlist)][self.steps]
        elif len(self.use_pv_list) > 1:
            use_pv = sum(np.array(self.config[pvlist[i]]) for i in self.use_pv_list)
        else:
            use_pv = self.config[pvlist[self.use_pv_list[0]]]
        self.pv_gen = use_pv
        return use_pv

    def _multi_user_model(self) -> object:
        multi_user = pyo.ConcreteModel("multi_user_multi_user")
        # ##multi_user setting param
        multi_user.users = pyo.RangeSet(0, self.num_agent - 1)
        multi_user.T_len = pyo.RangeSet(0, self.t_len-1)
        multi_user.use_flag = pyo.Var(multi_user.users, multi_user.T_len, domain=pyo.Binary)
        multi_user.use_power = pyo.Var(multi_user.users, multi_user.T_len, within=NonNegativeReals)
        multi_user.grid_use = pyo.Var(multi_user.users, multi_user.T_len, within=NonNegativeReals, bounds=(0, 3000))
        multi_user.solar_use = pyo.Var(multi_user.users, multi_user.T_len, within=NonNegativeReals)

        # info collection
        user_early_start = []
        user_late_end = []
        user_1t_use = []
        user_use_freq = []
        for i in range(self.num_agent):
            user_early_start.append(self.config[self.agents[i]][4])
            user_late_end.append(self.config[self.agents[i]][3])
            user_1t_use.append(self.config[self.agents[i]][0])
            user_use_freq.append(self.config[self.agents[i]][1])
        use_pv = self.pv_use()

        def t1_use_rule(m, us, t):
            return m.use_power[us, t] == user_1t_use[us] * m.use_flag[us, t]
        multi_user.t1_use = pyo.Constraint(multi_user.users, multi_user.T_len, rule=t1_use_rule)

        def fre_use_rule(m, us):
            return sum(m.use_flag[us, t] for t in m.T_len) == user_use_freq[us]
        multi_user.fre_use = pyo.Constraint(multi_user.users, rule=fre_use_rule)

        def early_start_rule(m, us):
            return sum(m.use_flag[us, t] for t in range(0, user_early_start[us])) <= 0
        multi_user.early_start = pyo.Constraint(multi_user.users, rule=early_start_rule)

        def late_stop_rule(m, us):
            return sum(m.use_flag[us, t] for t in range(user_late_end[us], self.t_len)) <= 0
        multi_user.late_stop = pyo.Constraint(multi_user.users, rule=late_stop_rule)

        def _solar_use_rule(m, t):
            return sum(m.solar_use[us, t] for us in m.users) <= use_pv[t]
        multi_user._solar_use = pyo.Constraint(multi_user.T_len, rule=_solar_use_rule)

        def total_use_rule(m, us, t):
            return m.solar_use[us, t] + m.grid_use[us, t] == m.use_power[us, t]
        multi_user.total_use_ = pyo.Constraint(multi_user.users, multi_user.T_len, rule=total_use_rule)

        def total_sum_rule(m, us):
            return sum(m.use_power[us, t] for t in m.T_len) == user_1t_use[us]*user_use_freq[us]
        multi_user.total_sum = pyo.Constraint(multi_user.users, rule=total_sum_rule)

        def obj_rule(multi_user):
            power_obj = sum(1*multi_user.grid_use[us, t] + 0*multi_user.solar_use[us, t] for us in multi_user.users for t in multi_user.T_len)
            return power_obj
        multi_user.obj = pyo.Objective(rule=obj_rule, sense=minimize)

        return multi_user


m = MultiUser()
model = m._multi_user_model()
opt = pyo.SolverFactory('gurobi', solver_io="python")
opt.options['NonConvex'] = 2
opt.options['Heuristics'] = 0.1
opt.options['MIPFocus'] = 1
opt.options['NumericFocus'] = 3
result = opt.solve(model).write()
model.solar_use.pprint()
model.grid_use.pprint()
MPlot(model,m)


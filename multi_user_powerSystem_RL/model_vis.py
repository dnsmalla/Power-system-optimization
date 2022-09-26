import pandas as pd
import numpy as np
import os
from pyomo.environ import value
import matplotlib.pyplot as plt
#import japanize_matplotlib


class MPlot():
    """
    class for plot optimize model output
    ----------
    Parameters
    ----------
    model : model
    inf_cls : class having model information
    """

    def __init__(self, model: object = None,inf_cls=None) -> None:
        self.plot_path = "./plot_ouput/"
        os.makedirs(self.plot_path, exist_ok=True)
        self.u_class = inf_cls
        self.clear_plot()
        self.t = len(model.T_len)
        self.us = len(model.users)
        self.user_plot(model)
        self.source_plot(model)
        # self.dam_plot(model)
        # self.branch_plot(model)
        # self.dam_dam_in_out_total_plot(model)
        # self.dam_dam_in_out_plot(model)

    def user_plot(self, model, flag=True):
        fig, ax = plt.subplots(self.us, figsize=(20, 10))
        for us in model.users:
            plt.rc('xtick', labelsize=14)
            plt.rc('ytick', labelsize=14)
            ax[us].set_xlabel("Time in 30 minute", fontdict={"fontsize": 18, "fontweight": "bold"})
            ax[us].set_ylabel("Power [W]", fontdict={"fontsize": 18, "fontweight": "bold"})
            gen = [round(value(model.use_power[us, t]), 2) for t in range(self.t)]
            gen_grid = [round(value(model.grid_use[us, t]), 2) for t in range(self.t)]
            gen_solar = [round(value(model.solar_use[us, t]), 2) for t in range(self.t)]
            total_use = np.array(gen_grid)+np.array(gen_solar)
            x_axis = list(range(self.t))
            ax[us].plot(x_axis, gen, "r-*", label="Use-"+str(us)+"[W]")
            ax[us].bar(x_axis, total_use, color="g", label="Solar use[W]")
            ax[us].bar(x_axis, gen_grid, color="b", label=" Grid use[W]")
            ax[us].set_xlim(-0.5, self.t-0.5)
            ax[us].set_title("User-"+str(us))
            h1, l1 = ax[us].get_legend_handles_labels()
            ax[us].legend(h1, l1, prop={'size': 16}, bbox_to_anchor=(0, 1), loc='upper left')
        fig.tight_layout()
        plt.savefig(self.plot_path+"user_info"+"-gen.png")
        plt.close()

    def source_plot(self, model, flag=True):
        fig, ax = plt.subplots(3, figsize=(20, 10))
        for s_n in range(3):
            plt.rc('xtick', labelsize=14)
            plt.rc('ytick', labelsize=14)
            ax[s_n].set_xlabel("Time in 30 minute", fontdict={"fontsize": 18, "fontweight": "bold"})
            ax[s_n].set_ylabel("Power [W]", fontdict={"fontsize": 18, "fontweight": "bold"})
            if s_n==0:
                gen_solar = [round(self.u_class.pv_gen[t], 2) for t in range(self.t)]
                use_solar = [sum(round(value(model.solar_use[us, t]), 2) for us in model.users)for t in range(self.t)]
                x_axis = list(range(self.t))
                ax[0].plot(x_axis, gen_solar, "r-*", label="Solar- generation"+"[W]")
                ax[0].bar(x_axis, use_solar, color="g", label="Solar use[W]")
                ax[0].set_xlim(-0.5, self.t-0.5)
                h1, l1 = ax[0].get_legend_handles_labels()
                ax[0].legend(h1, l1, prop={'size': 16}, bbox_to_anchor=(0, 1), loc='upper left')
            if s_n==1:
                gen_grid = [sum(round(value(model.grid_use[us, t]), 2)for us in model.users) for t in range(self.t)] 
                
                x_axis = list(range(self.t))
                ax[1].plot(x_axis, gen_grid, "r-*")
                ax[1].bar(x_axis, gen_grid, color="g", label="Grid use[W]")
                ax[1].set_xlim(-0.5, self.t-0.5)
                h1, l1 = ax[1].get_legend_handles_labels()
                ax[1].legend(h1, l1, prop={'size': 16}, bbox_to_anchor=(0, 1), loc='upper left')
            if s_n==2:
                use_power = np.array([[round(value(model.use_power[us, t]), 2) for t in range(self.t)]for us in model.users])
                use_p =pd.DataFrame(use_power.T,columns=["user-1","user-2","user-3"])
                x_axis = list(range(self.t))
                ax[2].plot(x_axis, np.sum(use_power,axis=0), "r-*")
                ax[2].set_xlim(-0.5, self.t-0.5)
                ax = use_p.plot(kind='bar', stacked=True, ax = ax[2],rot=0)

        fig.tight_layout()
        plt.savefig(self.plot_path+"Power-available.png")
        plt.close()

    
    def clear_plot(self):
        for file in os.scandir(self.plot_path):
            if file.path != self.plot_path+'.ipynb_checkpoints':
                os.remove(file.path)


if __name__ == "__main__":
    MPlot()

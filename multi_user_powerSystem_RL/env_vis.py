from typing import List
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


class EPlot():
    """
    class for plot optimize model output
    ----------
    Parameters
    ----------
    model : model
    inf_cls : class having model information
    """

    def __init__(self, actions: List=[],inf_cls=None) -> None:
        self.plot_path = "./plot_ouput/"
        os.makedirs(self.plot_path, exist_ok=True)
        self.actions = actions
        self.u_class = inf_cls
        self.clear_plot()
        self.t = len(self.actions[0])
        self.us = self.u_class.num_agent
        self.user_plot()
        self.source_plot()
        # self.dam_plot(model)
        # self.branch_plot(model)
        # self.dam_dam_in_out_total_plot(model)
        # self.dam_dam_in_out_plot(model)

    def user_plot(self):
        fig, ax = plt.subplots(self.us, figsize=(20, 10))
        for us in range(self.us):
            plt.rc('xtick', labelsize=14)
            plt.rc('ytick', labelsize=14)
            ax[us].set_xlabel("Time in 30 minute", fontdict={"fontsize": 18, "fontweight": "bold"})
            ax[us].set_ylabel("Power [W]", fontdict={"fontsize": 18, "fontweight": "bold"})
            power_use = [self.actions[us][t]*self.u_class.config[self.u_class.agents[us]][0] for t in range(self.t)]
            x_axis = list(range(self.t))
            ax[us].plot(x_axis, power_use, "r-*", label="Use-"+str(us)+"[W]")
            ax[us].bar(x_axis, power_use, color="g", label="Solar use[W]")
            ax[us].set_xlim(-0.5, self.t-0.5)
            ax[us].set_title("User-"+str(us))
            h1, l1 = ax[us].get_legend_handles_labels()
            ax[us].legend(h1, l1, prop={'size': 16}, bbox_to_anchor=(0, 1), loc='upper left')
        fig.tight_layout()
        plt.savefig(self.plot_path+"user_info"+"-gen.png")
        plt.close()

    def source_plot(self):
        fig, ax = plt.subplots(2, figsize=(20, 10))
        for s_n in range(2):
            plt.rc('xtick', labelsize=14)
            plt.rc('ytick', labelsize=14)
            ax[s_n].set_xlabel("Time in 30 minute", fontdict={"fontsize": 18, "fontweight": "bold"})
            ax[s_n].set_ylabel("Power [W]", fontdict={"fontsize": 18, "fontweight": "bold"})
            if s_n==0:
                gen_solar = [self.u_class.pv_gen[t] for t in range(self.t)]
                total_p = [sum(self.actions[us][t]*self.u_class.config[self.u_class.agents[us]][0] for us in range(self.us))for t in range(self.t)]
                x_axis = list(range(self.t))
                ax[0].plot(x_axis, gen_solar, "r-*", label="Solar- generation"+"[W]")
                ax[0].bar(x_axis, total_p, color="g", label="Solar use[W]")
                ax[0].set_xlim(-0.5, self.t-0.5)
                h1, l1 = ax[0].get_legend_handles_labels()
                ax[0].legend(h1, l1, prop={'size': 16}, bbox_to_anchor=(0, 1), loc='upper left')

        fig.tight_layout()
        plt.savefig(self.plot_path+"Power-available.png")
        plt.close()

    
    def clear_plot(self):
        for file in os.scandir(self.plot_path):
            if file.path != self.plot_path+'.ipynb_checkpoints':
                os.remove(file.path)


if __name__ == "__main__":
    EPlot()

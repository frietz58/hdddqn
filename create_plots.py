from utils.trajectory_plotting import plot_action_differences, plot_decomposed_q_vals
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    dirname = "checkpoints/2022.02.22-22:56:22/trajectories/2022.03.30-17:10:01"
    colors = [
        "#648FFF",
        "#785EF0",
        "#DC267F",
        "#FE6100",
        "#FFB000",
        "#365363"
    ]

    plt.rcParams["font.size"] = "17"
    plot_decomposed_q_vals(
        dirname,
        (0, 10000),  # interval of transitions to plot
        figsize=(8, 6),
        ylim_buffer=(-20, 10),
        ylim=(-135, 30),
        plot_dir=dirname,
        bar_width=0.3,
        level="",
        qval_key="q_vals",
        component_names_key="component_names",
        action_names_key="action_names",
        create_latex_table=False,
        fontsize=26,
        colors=[colors[2], colors[3], colors[-1]],
        legend=False
    )

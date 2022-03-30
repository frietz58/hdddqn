from utils.trajectory_plotting import plot_action_differences, plot_decomposed_q_vals
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    dirname = os.path.dirname(os.path.abspath(__file__))
    colors = [
        "#648FFF",
        "#785EF0",
        "#DC267F",
        "#FE6100",
        "#FFB000",
        "#365363"
    ]

    plt.rcParams["font.size"] = "17"
    action_differences = [
        {
            "time": 47,
            "comparisons": [
                ("South", "West"),
            ]
        },
        {
            "time": 50,
            "comparisons": [
                ("South", "East"),
            ]
        },
    ]
    for ad in action_differences:
        plt.rcParams["font.size"] = "15"
        plot_action_differences(
            dirname,
            ad["time"],
            action_comparisons=ad["comparisons"],
            figsize=(8, 4),
            ylim_buffer=(-1, 2),
            # colors=COMPONENT_COLORS["atomic"],
            plot_dir=dirname,
            # dpi=DPI,
            # component_linestyles=COMPONENT_LINESTYLES,
            qval_key="q_vals",
            component_names_key="component_names",
            action_names_key="action_names",
            fontsize=26,
            colors=[colors[2], colors[3], colors[-1]],
            legend=False
        )

    intervals = [(47, 48), (50, 51)]
    for interval in intervals:
        plot_decomposed_q_vals(
            dirname,
            interval,
            figsize=(8, 6),
            ylim_buffer=(-20, 10),
            ylim=(-115, 10),
            plot_dir=dirname,
            bar_width=0.3,
            # dpi=DPI,
            # colors=COMPONENT_COLORS["atomic"],
            # component_linestyles=COMPONENT_LINESTYLES,
            level="",
            qval_key="q_vals",
            component_names_key="component_names",
            action_names_key="action_names",
            create_latex_table=False,
            fontsize=26,
            colors=[colors[2], colors[3], colors[-1]],
            legend=False
        )

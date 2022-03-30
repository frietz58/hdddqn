import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import matplotlib
from utils import format_counter
from matplotlib.patches import Circle, PathPatch
from utils import latex_table_from_numpy, get_filtered_trace, list_npz_files
from utils import normed_entropy, softmax

# this fixes the minus sign being unknown in my custom latex font...
matplotlib.rcParams['axes.unicode_minus'] = False

# Add every font at the specified location
font_dir = ['/home/finn/Desktop/msc_thesis/non_code_plots/font']
for font in font_manager.findSystemFonts(font_dir):
    print(font)
    font_manager.fontManager.addfont(font)

plt.rcParams["font.size"] = "15"
plt.rcParams["font.family"] = "CMU Serif"

global DPI
DPI = 100

global COMPONENT_COLORS
COMPONENT_COLORS = {
    "atomic": ["tab:blue", "tab:orange", "tab:red", "tab:green", "tab:grey"],
    "meta": ["tab:purple", "tab:olive", "tab:brown"]
}

global COMPONENT_LINESTYLES
COMPONENT_LINESTYLES = ["solid", "dotted", "dashed", "dashdot", "dashdotdotted"]


def _plot_decomposed_reward(
        q_vals,
        labels,
        action_names,
        file_name="",
        mode="atomic",
        vertical_text_bar_space=3,
        x_tick_rotation=0,
        bar_width=0.2,
        ylim_buffer=(-50, 30),
        figsize=(12, 4),
        annotation_rotation=90,
        dpi=None,
        colors=None,
        component_linestyles=None,
        mean_subtracted=True,
        mean_subtracted_ylim_buffer=(-3, 3),
        mean_subtract_text_bar_space=0.5,
        shorten_legend=False,
        shorten_action_names=False,
        ylim=None,
        legend=True

):
    labels_done = []
    abs_component_bars = []
    abs_component_labels = []

    if shorten_action_names:
        new_action_names = []
        for a_name in action_names:
            if "location" in a_name:
                a_name = a_name.replace("location", "\nlocation")
            new_action_names.append(a_name)
        action_names = new_action_names

    if colors is None:
        colors = COMPONENT_COLORS[mode]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)

    for action_idx in range(0, q_vals.shape[1]):  # actions are columns
        for component_idx in range(0, q_vals.shape[0]):
            # we only want one legend entry per component...
            if shorten_legend:
                for comp_idx in range(len(labels)):
                    labels[comp_idx] = labels[comp_idx].replace("distance", "dist.")

            label = f'{labels[component_idx]}'

            if label in labels_done:
                label = ""
            else:
                labels_done.append(label)
                abs_component_labels.append(label)

            offset = (q_vals.shape[
                          0] - 1) / 2  # how much to move each component bar in the action group away from center
            offset_facs = np.arange(0, q_vals.shape[0]) - offset  # the concrete factors based on the offset

            if component_idx == q_vals.shape[0] - 1:  # range is zero based, shape not...
                color = colors[-1]  # last component is overall, take last color
            else:
                color = colors[component_idx]

            if component_linestyles is None:
                component_linestyles = COMPONENT_LINESTYLES

            if not mean_subtracted:
                bar_y = q_vals[component_idx][action_idx],
            else:
                component_mean = np.mean(q_vals[component_idx])
                bar_y = (q_vals[component_idx][action_idx]) - (component_mean)
                ylim_buffer = mean_subtracted_ylim_buffer
                vertical_text_bar_space = mean_subtract_text_bar_space

            # plot reward component bars
            rect = ax.bar(
                action_idx + (bar_width * offset_facs[component_idx]),  # the position for a bar in each group
                bar_y,
                bar_width,
                label=label,
                color=color,
                linestyle=component_linestyles[component_idx],
                linewidth=1.5,
                edgecolor="black"
            )
            abs_component_bars.append(rect)

            height = rect[0].get_height()
            if height > 0:
                text_y = height + vertical_text_bar_space
                va = "bottom"  # depending on pos or neg value set vertical alignment of text...
                rotation = annotation_rotation
            else:
                text_y = height - vertical_text_bar_space
                va = "top"
                rotation = annotation_rotation

            ax.text(
                rect[0].get_x() + rect[0].get_width() / 2.,
                text_y,
                np.around(height, 2),
                ha='center',
                va=va,
                rotation=rotation
            )

    # set x tick labels
    if not action_names:
        xtick_labels = [f"A{idx}" for idx in range(0, q_vals.shape[1])]
    else:
        xtick_labels = action_names

    xtick_labels = [l.replace("-", "-\n") for l in xtick_labels]

    a_vals = np.around(q_vals[-1, :], 3)
    best_action_labels = [""] * q_vals.shape[1]
    # best_action_labels[np.argmax(a_vals)] = "%".replace("%", str(max(a_vals)))  # currently I dont want best value anno
    # xtick_labels[np.argmax(a_vals)] = "$\mathbf{%}$".replace("%", xtick_labels[np.argmax(a_vals)])

    final_xtick_labels = []
    for idx in range(0, q_vals.shape[1]):
        # final_xtick_labels.append(f'{xtick_labels[idx]}\n{a_vals[idx]}\n{best_action_labels[idx]}')
        # final_xtick_labels.append(f'{xtick_labels[idx]}\n{best_action_labels[idx]}')
        final_xtick_labels.append(f'{xtick_labels[idx]}')
    ax.set_xticklabels(final_xtick_labels)
    ax.set_xticks(range(0, q_vals.shape[1]))
    ax.xaxis.set_tick_params(rotation=x_tick_rotation)

    # get and update the y_margin so that bar height text isnt outside of axis...
    x_marg, y_marg = ax.margins()
    ax.margins(x_marg, 0.07)

    # update ylim that we always have some padding at the top...
    if ylim is None:
        y_lim = ax.get_ylim()
        new_ylim = (y_lim[0] + ylim_buffer[0], y_lim[1] + ylim_buffer[1])
        ax.set_ylim(new_ylim)
    else:
        ax.set_ylim(ylim)

    plt.xlabel("Actions")
    plt.ylabel("State-action value")

    if legend:
        plt.legend(
            bbox_to_anchor=(0, 1.02, 1, 0.2),
            loc="lower left",
            borderaxespad=0,
            mode="expand",
            ncol=q_vals.shape[0],
            fancybox=True,
            shadow=True
        )

    # plt.draw()
    plt.tight_layout()

    if dpi is None:
        dpi = DPI

    if file_name:
        trace = get_filtered_trace()
        plt.savefig(file_name, bbox_inches="tight", dpi=dpi, metadata={"trace": trace})
        plt.close()

    new_fig = plt.figure(figsize=figsize)
    h, l = fig.gca().get_legend_handles_labels()
    new_fig.legend(h, l, ncol=q_vals.shape[0], fancybox=True, shadow=True, labelspacing=5)
    plt.savefig(
        os.path.join(
            os.path.dirname(file_name),
            "decomposedReward_legend.png"
        ),
        bbox_inches="tight",
        dpi=dpi,
        metadata={"trace": trace}
        )


    return abs_component_bars, abs_component_labels


def plot_decomposed_q_vals(
        trajectory_dir,
        interval=(0, 10),
        level="atomic",
        figsize=(12, 4),
        ylim_buffer=(-50, 30),
        entropy_weighted=False,
        plot_dir=None,
        dpi=None,
        colors=None,
        component_linestyles=None,
        shorten_legend=False,
        shorten_action_names=False,
        bar_width=0.2,
        text_bar_space=3,
        time_key="overall_counter",
        qval_key="",
        component_names_key="",
        action_names_key='',
        create_latex_table=True,
        ylim=None,
        fontsize=None,
        legend=True,
):
    files = os.listdir(trajectory_dir)
    npz_files = []

    for file in files:
        if ".npz" in file:
            npz_files.append(file)

    npz_files.sort()

    for file in npz_files:
        file_path = os.path.join(trajectory_dir, file)
        data = np.load(file_path, allow_pickle=True)

        if not qval_key:
            q_val_key = f"{level}_q_vals" if not entropy_weighted else f"{level}_inverse_entropy_weighted_q_vals"
        else:
            q_val_key = qval_key

        if data[time_key] in range(interval[0], interval[1]):
            if level in file or not level:
                for mean_subtracted in [False]:
                    q_vals = data[q_val_key]
                    # data["atomic_softmax_mat"]
                    # data["atomic_component_entropies"]
                    if not component_names_key:
                        component_names = list(data[f"{level}_component_names"]),
                    else:
                        component_names = data[component_names_key]
                    if not action_names_key:
                        action_names = list(data[f"{level}_action_names"]),
                    else:
                        action_names = data[action_names_key]

                    print(f"Overall time: {data[time_key]}")
                    print("Qvals:\n", q_vals)
                    print("Component means:", np.mean(q_vals, axis=1))
                    print("Component STDs:", np.std(q_vals, axis=1))
                    print("Components:", component_names)
                    print("Actions:", action_names)
                    print()

                    # this is very weird, I get a tuple of arrays when I load it, although it should just be an array
                    if type(component_names) == tuple:
                        component_names_flat = []
                        for sublist in component_names:
                            component_names_flat.extend(sublist)
                    else:
                         component_names_flat = component_names

                    if type(component_names) == tuple:
                        action_names_flat = []
                        for sublist in action_names:
                            action_names_flat.extend(sublist)
                    else:
                        action_names_flat = action_names

                    file_name = "episode:" + file.split("_episode:")[1].split("_")[0] + "_overallCounter:"  + format_counter(data[time_key]) + "_plot"  # god this is so ugly
                    file_name += f"_{level}" if level else ""
                    if entropy_weighted:
                        file_name += "_entropyWeighted"

                    if mean_subtracted:
                        file_name += "_meanSubtracted"

                    if plot_dir is None:
                        plot_dir = PLOT_DIR

                    if dpi is None:
                        dpi = DPI

                    if colors is None:
                        if level:
                            colors = COMPONENT_COLORS[level]
                        else:
                            colors = COMPONENT_COLORS["atomic"]


                    if component_linestyles is None:
                        component_linestyles = COMPONENT_LINESTYLES

                    file_name = os.path.join(plot_dir, file_name)

                    if fontsize:
                        plt.rcParams["font.size"] = str(fontsize)
                    _plot_decomposed_reward(
                        q_vals=q_vals,
                        labels=list(component_names_flat),
                        action_names=list(action_names_flat),
                        file_name=file_name,
                        figsize=figsize,
                        x_tick_rotation=0,
                        ylim_buffer=ylim_buffer,
                        dpi=DPI,
                        colors=colors,
                        component_linestyles=component_linestyles,
                        mean_subtracted=mean_subtracted,
                        shorten_legend=shorten_legend,
                        shorten_action_names=shorten_action_names,
                        bar_width=bar_width,
                        vertical_text_bar_space=text_bar_space,
                        ylim=(-13, 13) if mean_subtracted else ylim,
                        legend=legend
                    )

                    # create latex table
                    if create_latex_table:
                        file_name += ".tex"
                        latex_table_from_numpy(
                            q_vals,
                            row_labels=["\\acrshort{gdc}", "\\acrshort{sdc}", "\\acrshort{ddc}", "Sum"],
                            col_labels=action_names_flat,
                            file_name=os.path.join(plot_dir, file_name)
                        )


def plot_action_differences(
        trajectory_dir,
        time,
        action_comparisons=[("Stop", "North-West")],
        level="atomic",
        figsize=(12, 8),
        annotation_rotation=0,
        vertical_text_bar_space=0.2,
        ylim_buffer=(-3, 3),
        entropy_weighted=False,
        colors=None,
        plot_dir=None,
        dpi=None,
        component_linestyles=None,
        time_key="overall_counter",
        qval_key=None,
        component_names_key=None,
        action_names_key=None,
        fontsize=None,
        legend=True

):
    files = os.listdir(trajectory_dir)
    npz_files = []

    if fontsize:
        plt.rcParams["font.size"] = str(fontsize)

    for file in files:
        if ".npz" in file:
            npz_files.append(file)

    npz_files.sort()

    fig, axs = plt.subplots(len(action_comparisons), 1, figsize=figsize)

    if colors is None:
        colors = COMPONENT_COLORS[level]
    labels_done = []
    abs_component_bars = []
    abs_component_labels = []
    if qval_key is None:
        q_val_key = f"{level}_q_vals" if not entropy_weighted else f"{level}_inverse_entropy_weighted_q_vals"
    else:
        q_val_key = qval_key

    for file in npz_files:
        file_path = os.path.join(trajectory_dir, file)
        data = np.load(file_path, allow_pickle=True)

        if data[time_key] == time:
            q_vals = data[q_val_key]
            if component_names_key is None:
                component_names = list(data[f"{level}_component_names"]),
            else:
                component_names = data[component_names_key]
            if action_names_key is None:
                action_names = list(data[f"{level}_action_names"]),
            else:
                action_names = data[action_names_key]

            # this is very weird, I get a tuple of arrays when I load it, although it should just be an array
            if type(component_names) == tuple:
                component_names_flat = []
                for sublist in component_names:
                    component_names_flat.extend(sublist)
            else:
                component_names_flat = component_names

            if type(action_names) == tuple:
                action_names_flat = []
                for sublist in action_names:
                    action_names_flat.extend(sublist)
            else:
                action_names_flat = action_names

            comp_counter = 0
            for tup in action_comparisons:
                a0, a1 = tup

                # get indices of desired actions
                a0_idx = list(action_names_flat).index(a0)
                a1_idx = list(action_names_flat).index(a1)

                # get reward vecs
                a0_reward_vec = q_vals[:, a0_idx]
                a1_reward_vec = q_vals[:, a1_idx]

                # calc diff
                diff = a0_reward_vec - a1_reward_vec
                if type(axs) is np.ndarray:
                    ax = axs[comp_counter]
                else:
                    ax = axs  # if we only plot the RDX for one comparion...
                for component_idx, val in enumerate(diff):

                    # horizontal line at 0
                    ax.axhline(0, c="black", linewidth=0.5)

                    if component_idx == q_vals.shape[0] - 1:  # range is zero based, shape not...
                        color = colors[-1]  # last component is overall, take last color
                    else:
                        color = colors[component_idx]

                    if component_names_flat[component_idx] not in labels_done:
                        label = component_names_flat[component_idx]
                    else:
                        label = ""

                    if component_linestyles is None:
                        component_linestyles = COMPONENT_LINESTYLES

                    rect = ax.bar(
                        component_idx,  # the position for a bar in each group
                        val,
                        0.8,
                        label=component_names_flat[component_idx],
                        color=color,
                        linestyle=component_linestyles[component_idx],
                        linewidth=1.5,
                        edgecolor="black"
                    )

                    abs_component_bars.append(rect)

                    height = rect[0].get_height()
                    if height > 0:
                        text_y = height + vertical_text_bar_space
                        va = "bottom"  # depending on pos or neg value set vertical alignment of text...
                        rotation = annotation_rotation
                    else:
                        text_y = height - vertical_text_bar_space
                        va = "top"
                        rotation = annotation_rotation

                    ax.text(
                        rect[0].get_x() + rect[0].get_width() / 2.,
                        text_y,
                        np.around(height, 2),
                        ha='center',
                        va=va,
                        rotation=rotation
                    )

                # get and update the y_margin so that bar height text isnt outside of axis...
                x_marg, y_marg = ax.margins()
                ax.margins(x_marg, 0.07)

                # update ylim that we always have some padding at the top...
                y_lim = ax.get_ylim()
                # new_ylim = (min(y_lim[0] + ylim_buffer[0], -11), max(y_lim[1] + ylim_buffer[1], 11))
                new_ylim = (min(y_lim[0] + ylim_buffer[0], 0), min(y_lim[1] + ylim_buffer[1], 15))
                ax.set_ylim(new_ylim)

                # only plot legend once
                if comp_counter == 0:
                    if legend:
                        ax.legend(
                            bbox_to_anchor=(0, 1.25, 1, 0.2),
                            loc="lower left",
                            borderaxespad=0,
                            mode="expand",
                            ncol=q_vals.shape[0],
                            fancybox=True,
                            shadow=True
                        )

                # set title
                ax.set_title(f"{a0} vs. {a1}")
                # ax.set_ylabel("(s, a, b)")

                ax.tick_params(
                    axis='x',  # changes apply to the x-axis
                    which='both',  # both major and minor ticks are affected
                    bottom=False,  # ticks along the bottom edge are off
                    top=False,  # ticks along the top edge are off
                    labelbottom=False)  # labels along the bottom edge are off

                comp_counter += 1

            plt.tight_layout()

            if plot_dir is None:
                plot_dir = PLOT_DIR

            if dpi is None:
                dpi = DPI

            file_name = f"plot_rdx_time{time}_{action_comparisons}{'_entropyWeighted' if entropy_weighted else ''}"
            plt.savefig(os.path.join(plot_dir, file_name), bbox_inches="tight", dpi=dpi)



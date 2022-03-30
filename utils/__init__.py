import os
from datetime import datetime
import torch
import numpy as np
import random
import yaml
import traceback
from scipy.stats import multivariate_normal


def list_npz_files(dir):
    files = os.listdir(dir)
    npz_files = []

    for file in files:
        if ".npz" in file:
            npz_files.append(file)

    npz_files.sort()
    return npz_files


def get_filtered_trace(key0="finn", key1="5rietz"):
    trace_list = traceback.format_stack()
    filtered_trace = []
    for item in trace_list:
        if key0 in item or key1 in item:

            filtered_trace.append(item.split("\n")[0])

    filtered_trace = "  --> ".join(filtered_trace)
    print(filtered_trace)
    return filtered_trace


def setup_checkpoint_dir(dir=".", postfix=""):
    checkpoint_str = datetime.now().strftime("%Y.%m.%d-%H:%M:%S") + postfix
    checkpoint_dir = os.path.join(dir, checkpoint_str)

    if not os.path.exists(dir):
        os.mkdir(dir)

    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    return checkpoint_dir, checkpoint_str


def set_seeds(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def save_yaml(data_dict, path, exclude=[], filename="param_configuration.yaml"):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=False)  # make dir or raise alert if it already exists (should not happen)

    save_dict = dict(data_dict)  # make shallow copy so that we don't modify the existing dict
    if exclude:
        for key in exclude:
            if key in save_dict.keys():
                save_dict.pop(key)

    with open(os.path.join(path, filename), "w") as f:
        yaml.dump(save_dict, f)


def min_max_norm(val, min, max):
    return (val - min)/(max - min)


def get_goal_vec_rep(two_d_goal, youbot):
    """
    Returns an agent-centric vector representation of a navigation goal.
    A navigation goal coordinate pair is represented with a vector containing sine and cosine of the angle of the
    vector between the agent and the goal, as well as the length of that vector.
    :param two_d_goal: The x,y coordinates of the navigation goal
    :return: Vector of length three containing the above described components
    """
    # get vec between agent and goal
    you_pos = youbot.get_position()[0:2]
    dir_vec = np.array([two_d_goal[0] - you_pos[0], two_d_goal[1] - you_pos[1]])

    # get angle (direction) of that vector and the angle sine and cosine
    vec_dir = np.arctan2(dir_vec[1], dir_vec[0])
    dir_sine = np.sin(vec_dir)
    dir_cos = np.cos(vec_dir)

    # get length of the vector
    vec_len = np.linalg.norm(dir_vec)

    return np.array([dir_sine, dir_cos, vec_len])


def softmax(x_discrete):
    # https://stackoverflow.com/questions/54880369/implementation-of-softmax-function-returns-nan-for-high-inputs
    f = np.exp(x_discrete - np.max(x_discrete))  # shift values
    return f / f.sum(axis=0)


def normed_entropy(dist):
    softmax_log = np.log2(dist)
    ex = [-p_x * softmax_log[idx] for idx, p_x in enumerate(dist)]
    entropy = sum(ex)
    normed_entropy = entropy / np.log2(dist.shape[0])  # normalize
    return normed_entropy


def format_counter(num):
    return "{:05}".format(num)


def latex_table_from_numpy(np_arr, row_labels=[], col_labels=[], file_name=""):
    np_arr = np.around(np_arr, 2)
    table_str = "\\begin{figure}[t]\n"
    table_str += "\\begin{adjustbox}{center}\n"
    table_str += "\\begin{tabular}{@{}r$@{}}\n\\toprule\n".replace("$", "c"*len(col_labels))
    col_labels.insert(0, "Comp.")

    col_labels = ["\\thead{$}".replace("$", l) for l in col_labels]
    col_labels = [l.replace("-", "- \\\\ ") for l in col_labels]

    table_str += " & ".join(col_labels) + " \\\\\n"
    table_str += "\\midrule\n"
    for row in range(np_arr.shape[0]):
        
        # find best value in row for highlighting
        best_val_idx = np.argmax(np_arr[row, :])
        best_val = np_arr[row, best_val_idx]
        best_val_bf_str = "$\\mathbf{&}$".replace("&", str(best_val))

        row_str = row_labels[row] + " & " + " & ".join(map(str, np_arr[row, :])) + " \\\\\n"
        row_str = row_str.replace(str(best_val), best_val_bf_str)
        table_str += row_str
    
    table_str += "\\bottomrule\n"
    table_str += "\\end{tabular}\n"
    table_str += "\\end{adjustbox}\n"
    table_str += "%\caption{}\n"
    table_str += "%\label{}\n"
    table_str += "\\end{figure}\n"
    print(table_str)

    if file_name:
        with open(file_name, "w") as f:
           f.write(table_str)

    # print(" \\\\\n".join([" & ".join(map(str, line)) for line in np_arr]))


def to_tensor(data, device):
    if not torch.is_tensor(data):
        return torch.tensor(data, device=device).double()
    else:
        return data.double()
        

def gaussian_activation(x, y, xmean, ymean, x_var=1, xy_cov=0, yx_cov=0, y_var=1):
    """
    Return the value for a 2d gaussian distribution with mean at [xmean, ymean] and the covariance matrix based on
    [[x_var, xy_cov],[yx_cov, y_var]].
    """
    means = [xmean, ymean]
    cov_mat = [
        [x_var, xy_cov],
        [yx_cov, y_var]
    ]

    rv = multivariate_normal(means, cov_mat)

    return rv.pdf([x, y])





import datetime
import os
from collections import OrderedDict
import argparse
from yaml import load
import torch.optim as optim
import importlib
import torch.nn.functional as F

from utils import setup_checkpoint_dir
from utils import save_yaml
from utils import set_seeds
from decomposed_agent import Agent



def run_repetitions(params, env, atomic_model_class, meta_model_class, n_repetitions=2, postfix=""):
    abs_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "checkpoints")
    cp_dir, cp_str = setup_checkpoint_dir(abs_path, postfix=postfix)

    conf_dir = cp_dir
    save_yaml(params, conf_dir)

    # set_seeds(params["seed"])

    # execute conf!
    for repetition in range(n_repetitions):
        cp_rep_dir = os.path.join(conf_dir, str(repetition))
        if not os.path.exists(cp_rep_dir):
            os.makedirs(cp_rep_dir)

        agent = Agent(atomic_model_class, meta_model_class, env, params, cp_rep_dir, cp_str, atomic_save_freq=5000, meta_save_freq=100)
        agent.train(atomic_training_steps=params["atomic_total_training_steps"], meta_training_steps=params["meta_total_training_steps"])
        del agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-cl', dest='config_list', nargs='+', required=False,
        help="List of YAML config files to run multiple experiments, with different hyperparameters, one after another.",
        metavar=("path/to/config_1.yaml", "/path/to/config_2.yaml"))
    args = parser.parse_args()

    # define 'default 'hyperparameter space
    params = OrderedDict()
    params["atomic_total_training_steps"] = 100000  # train the atomic controller for at least this many steps, but continue until meta controller is done
    params["atomic_eps_decay"] = 0.4  # The fraction of all interactions we want epsilon to decay over
    params["atomic_eps_start"] = 1  # initial value for epsilon greedy
    params["atomic_eps_end"] = 0.1  # final epsilon value
    params["atomic_batch_size"] = 1024  # transition batch size
    params["atomic_policy_update_rate"] = 1  # how many gradient steps per update
    params["atomic_gamma"] = 0.9  # discount factor
    params["atomic_use_per"] = True  # use per 
    params["atomic_target_update_rate"] = 250000  # every this many transitions, do a hard update of target network
    params["atomic_model_file"] = "models/medium/model.py"  # which model file to use

    # to continue with a previous CP:
    params["atomic_cp_dir"] = ""  # dir containing .pth checkpoint files matching given model architecture  
    params["atomic_cp_str"] = ""  # string to identify which .pth file to load
    if params["atomic_cp_dir"]:
        # adjust we use pretrained weights, explortation can be adjusted...
        params["atomic_eps_decay"] = 0.1
        params["atomic_eps_start"] = 0.1
        params["atomic_eps_end"] = 0.1

    params["atomic_pretrain_steps"] = 150000  # train the atomic policy for this many steps before we start training meta policy
    params["meta_cp_dir"] = ""  # to continue with previous CP
    params["meta_cp_str"] = ""
    params["meta_total_training_steps"] = 50000  # the max number of env interactions for atomic policy
    params["meta_eps_decay"] = 0.4  # The fraction of all interactions we want epsilon to decay over
    params["meta_eps_start"] = 1  # initial value for epsilon greedy
    params["meta_eps_end"] = 0.1  # final epsilon value
    params["meta_batch_size"] = 256  # transition batch size
    params["meta_policy_update_rate"] = 3  # how many gradient steps per update
    params["meta_gamma"] = 1  # discount factor
    params["meta_target_update_rate"] = 1000  # every this many transitions, do a hard update of target network
    params["meta_use_per"] = True  # use per
    params["meta_model_file"] = "models/medium/model.py"  #  which model file to use

    params["optimization_steps"] = 1  
    params["policy_loss_fn"] = F.smooth_l1_loss
    params["mem_size"] = 1000000
    params["lr"] = 0.0005
    params["optimizer"] = optim.RMSprop
    # params["seed"] = 1337
    params["experiment_name"] = "default"

    params["env_file"] = "envs/opposite-both-required-hierarchical-decomposed-same-state-nav-env/simple-decomposed-nav-env.py"

    if args.config_list:
        print("Got config args")
        path_inserts = []
        for config_file in args.config_list:
            yaml_file = open(config_file)
            config = load(yaml_file)

            if config is not None:
                # either use from config or the one specified manually
                try:
                    env_file = config["env_file"]
                except KeyError:
                    print("No env path found in given config, using default values specified in script...")

                try:
                    meta_model_file = config["meta_model_file"]
                except KeyError:
                    print("No META model file found in given config, using default values specified in script...")

                try:
                    atomic_model_file = config["atomic_model_file"]
                except KeyError:
                    print("No ATOMIC model file found in given config, using default values specified in script...")

                for config_param in config.keys():
                    params[config_param] = config[config_param]

                # import env
                print(f"Env path to insert: {env_file}")
                env_spec = importlib.util.spec_from_file_location("simple-nav-env", env_file)
                env_module = importlib.util.module_from_spec(env_spec)
                env_spec.loader.exec_module(env_module)
                env = env_module.CoppeliaYouBotNavEnv(headless=True)

                params["env_path"] = env_file
                params["coppeliaScene"] = env.scene_file
                print(f"Using env: {env.scene_file}")

                # import models
                meta_model_spec = importlib.util.spec_from_file_location("model", meta_model_file)
                meta_model_module = importlib.util.module_from_spec(meta_model_spec)
                meta_model_spec.loader.exec_module(meta_model_module)
                meta_model = meta_model_module.QNetwork
                params["meta_model_path"] = meta_model_file
                print(f"Using META model: {meta_model_file}")

                atomic_model_spec = importlib.util.spec_from_file_location("model", atomic_model_file)
                atomic_model_module = importlib.util.module_from_spec(atomic_model_spec)
                atomic_model_spec.loader.exec_module(atomic_model_module)
                atomic_model = atomic_model_module.QNetwork
                params["atomic_model_path"] = atomic_model_file
                print(f"Using ATOMIC model: {atomic_model_file}")

                starting_time = datetime.datetime.now()

                run_repetitions(params, env, atomic_model, meta_model, 1, postfix=params["experiment_name"])
                env.close()
                del env
                del atomic_model
                del meta_model

                ending_time = datetime.datetime.now()
                exp_duration = ending_time - starting_time
                print(f"Experiment start: {starting_time}\nExperiment end: {ending_time}\nDuration: {exp_duration}")
            else:
                print(f"Given config '{config_file}' appears to be empty, not running experiment for it...")

        else:
            print("Exhausted given experiment list, exiting...")

    else:
        print("No configurations given via cli arg, using default params specified in script...")
        starting_time = datetime.datetime.now()

        # import env
        print(f"Env path to insert: {params['env_file']}")
        env_spec = importlib.util.spec_from_file_location("simple-nav-env", params['env_file'])
        env_module = importlib.util.module_from_spec(env_spec)
        env_spec.loader.exec_module(env_module)
        env = env_module.CoppeliaYouBotNavEnv(headless=True)

        params["coppeliaScene"] = env.scene_file
        print(f"Using env: {env.scene_file}")

        # import models
        meta_model_file = params["meta_model_file"]
        meta_model_spec = importlib.util.spec_from_file_location("model", meta_model_file)
        meta_model_module = importlib.util.module_from_spec(meta_model_spec)
        meta_model_spec.loader.exec_module(meta_model_module)
        meta_model = meta_model_module.QNetwork
        params["meta_model_path"] = meta_model_file
        print(f"Using META model: {meta_model_file}")

        atomic_model_file = params["atomic_model_file"]
        atomic_model_spec = importlib.util.spec_from_file_location("model", atomic_model_file)
        atomic_model_module = importlib.util.module_from_spec(atomic_model_spec)
        atomic_model_spec.loader.exec_module(atomic_model_module)
        atomic_model = atomic_model_module.QNetwork
        params["atomic_model_path"] = atomic_model_file
        print(f"Using ATOMIC model: {atomic_model_file}")

        run_repetitions(params, env, atomic_model, meta_model, 1, postfix=params["experiment_name"])
        env.close()

        ending_time = datetime.datetime.now()
        exp_duration = ending_time - starting_time
        print(f"Experiment start: {starting_time}\nExperiment end: {ending_time}\nDuration: {exp_duration}")

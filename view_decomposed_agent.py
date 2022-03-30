import argparse
import os
import importlib
import sys
import numpy as np
import torch
import yaml
from yaml import load
from decomposed_policy import Policy
from datetime import datetime
import matplotlib.pyplot as plt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-rd', '--record_data', action='store_true', dest='record_data')
    parser.set_defaults(record_data=False)
    args = parser.parse_args()

    # init agent and env from CP
    CHECKPOINT = "checkpoints/2022.02.22-22:56:22"
    ATOMIC_CHECKPOINT_IDENTIFIER_STR = "best_mean"
    META_CHECKPOINT_IDENTIFIER_STR = "best_mean"

    if args.record_data:
        save_dir = os.path.join(CHECKPOINT, "trajectories", datetime.now().strftime("%Y.%m.%d-%H:%M:%S"))
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    cp_dir = CHECKPOINT

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    yaml_file = open(os.path.join(CHECKPOINT, "param_configuration.yaml"))
    params = load(yaml_file, Loader=yaml.FullLoader)

    try:
        env_file = params["env_file"]
    except AttributeError:
        print("CP yaml does not contains env path! (old CP dir?)")
        sys.exit()

    sys.path.insert(0, os.path.dirname(env_file))
    env_spec = importlib.util.spec_from_file_location("simple-nav-env", env_file)
    env_module = importlib.util.module_from_spec(env_spec)
    env_spec.loader.exec_module(env_module)
    env = env_module.CoppeliaYouBotNavEnv(headless=True)  # instance needed for attributes

    all_files = os.listdir(CHECKPOINT)

    atomic_model_path = params["atomic_model_path"]
    atomic_model_spec = importlib.util.spec_from_file_location("model", atomic_model_path)
    atomic_model_module = importlib.util.module_from_spec(atomic_model_spec)
    atomic_model_spec.loader.exec_module(atomic_model_module)
    atomic_model_class = atomic_model_module.QNetwork

    atomic_policy = Policy(
        model_class=atomic_model_class,
        env=env,
        level="atomic",
        optimizer=params["optimizer"],
        device=device,
        checkpoint_dir="",
        gamma=params["atomic_gamma"],
        batch_size=params["atomic_batch_size"],
        lr=params["lr"],
        target_net_update_freq=params["atomic_target_update_rate"],
        restore_dir=CHECKPOINT,
        restore_str=ATOMIC_CHECKPOINT_IDENTIFIER_STR,  # used to determine concrete checkpoint
        tb_writer=None
    )
    meta_model_path = params["meta_model_path"]
    meta_model_spec = importlib.util.spec_from_file_location("model", meta_model_path)
    meta_model_module = importlib.util.module_from_spec(meta_model_spec)
    meta_model_spec.loader.exec_module(meta_model_module)
    meta_model_class = meta_model_module.QNetwork
    meta_policy = Policy(
        model_class=meta_model_class,
        env=env,
        level="meta",
        optimizer=params["optimizer"],
        device=device,
        checkpoint_dir="",
        gamma=params["meta_gamma"],
        batch_size=params["meta_batch_size"],
        lr=params["lr"],
        target_net_update_freq=params["meta_target_update_rate"],
        restore_dir=CHECKPOINT,
        restore_str=META_CHECKPOINT_IDENTIFIER_STR,  # used to determine concrete checkpoint
        tb_writer=None
    )
    env.close()

    env = env_module.CoppeliaYouBotNavEnv(headless=False)
    with torch.no_grad():

        overall_counter = 0

        for i in range(10):
            meta_episode_counter = 0
            meta_episode_rewards = []
            meta_episode_successes = []
            meta_step_counter = 0
            meta_episode_reward = np.zeros([env.n_reward_components["meta"]])
            meta_done = False
            meta_episode_success = False
            meta_obs = env.meta_reset()
            # env.agent.set_position([-3.35, -2, env.agent.get_position()[-1]])
            # meta_obs = env.get_meta_obs()

            while not meta_done:
                meta_action, stacked_meta_pred = meta_policy.select_action(meta_obs, epsilon=0, return_components=True)
                meta_pred = stacked_meta_pred.detach().cpu().numpy()
                env.do_meta_action(meta_action.cpu().detach().numpy().item())

                atomic_episode_counter = 0
                atomic_episode_rewards = []
                atomic_episode_successes = []
                atomic_step_counter = 0
                atomic_episode_reward = np.zeros([env.n_reward_components["atomic"]])
                atomic_done = False
                atomic_episode_success = False
                atomic_obs = env.atomic_reset()

                if args.record_data:
                    data_name = f"atomic:'{ATOMIC_CHECKPOINT_IDENTIFIER_STR}'_meta:'{META_CHECKPOINT_IDENTIFIER_STR}'_episode:{i}_overallCounter:{overall_counter}_metaCounter:{meta_step_counter}_atomicCounter:{atomic_step_counter}"
                    np.savez(
                        os.path.join(save_dir, data_name),
                        overall_counter=overall_counter,
                        obs=meta_obs,
                        q_vals=meta_pred,
                        agent_xy=env.agent.get_position()[0:2],
                        component_names=env.component_names["meta"],
                        level="meta"
                    )
                    img = env.vision_sensor.capture_rgb()
                    plt.imsave(os.path.join(save_dir, f"episode:{i}_overallCounter:{overall_counter}_metaCounter:{meta_step_counter}_atomicCounter:{atomic_step_counter}.png"), img)


                while not atomic_done:
                    action, stacked_pred = atomic_policy.select_action(atomic_obs, epsilon=0, return_components=True)
                    pred = stacked_pred.detach().cpu().numpy()
                    if args.record_data:
                        data_name = f"atomic:'{ATOMIC_CHECKPOINT_IDENTIFIER_STR}'_meta:'{META_CHECKPOINT_IDENTIFIER_STR}'_episode:{i}_overallCounter:{overall_counter}_metaCounter:{meta_step_counter}_atomicCounter:{atomic_step_counter}"
                        summed_pred = np.sum(pred, 0)  # calc overall action values
                        pred = np.append(pred, [summed_pred], 0)  # append it
                        np.savez(
                            os.path.join(save_dir, data_name),
                            overall_counter=overall_counter,
                            obs=atomic_obs,
                            q_vals=pred,
                            agent_xy=env.agent.get_position()[0:2],
                            component_names=env.component_names["atomic"] + ["Overall"],
                            action_names=env.atomic_action_names,
                            level="meta"
                        )
                        img = env.vision_sensor.capture_rgb()
                        plt.imsave(os.path.join(save_dir, f"episode:{i}_overallCounter:{overall_counter}_metaCounter:{meta_step_counter}_atomicCounter:{atomic_step_counter}.png"), img)

                    new_atomic_obs, reward, atomic_done, msg_dict, episode_success = env.atomic_step(action.cpu().detach().numpy().item())
                    atomic_episode_reward += reward
                    atomic_step_counter += 1
                    overall_counter += 1

                    # move to new state
                    atomic_obs = new_atomic_obs

                # atomic episode done
                atomic_episode_rewards.append(sum(atomic_episode_reward))
                atomic_episode_successes.append(int(episode_success))
                atomic_episode_counter += 1

                meta_new_obs, meta_reward, meta_done, meta_episode_success, meta_info = env.eval_meta_action()
                meta_episode_reward += meta_reward
                meta_obs = meta_new_obs
                meta_step_counter += 1
                overall_counter += 1

            # meta episode done
            meta_episode_rewards.append(sum(meta_episode_reward))
            meta_episode_successes.append(int(meta_episode_success))
            meta_episode_counter += 1

    env.close()




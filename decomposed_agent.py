from decomposed_policy import Policy
import torch
from torch.utils.tensorboard import SummaryWriter
from tensorboard import program
import random

import numpy as np


class Agent:
    def __init__(self, atomic_model_class, meta_model_class, env, params, cp_dir, cp_str, atomic_save_freq=500, meta_save_freq=50):
        self.atomic_model_class = atomic_model_class
        self.meta_model_class = meta_model_class
        self.env = env
        self.hyperparams = params
        self.cp_dir = cp_dir
        self.cp_str = cp_str
        self.experiment_name = cp_str
        self.atomic_safe_freq = atomic_save_freq
        self.meta_safe_freq = meta_save_freq
        self.atomic_pretrain_steps = params["atomic_pretrain_steps"]

        self.tb_writer = SummaryWriter(cp_dir, comment="TensorBoardData")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.atomic_policy = Policy(
            model_class=self.atomic_model_class,
            env=self.env,
            level="atomic",
            optimizer=params["optimizer"],
            device=self.device,
            checkpoint_dir=self.cp_dir,
            gamma=params["atomic_gamma"],
            batch_size=params["atomic_batch_size"],
            lr=params["lr"],
            target_net_update_freq=params["atomic_target_update_rate"],
            tb_writer=self.tb_writer,
            use_per=params["atomic_use_per"],
            restore_dir=params["atomic_cp_dir"],
            restore_str=params["atomic_cp_str"],
        )

        self.meta_policy = Policy(
            model_class=self.meta_model_class,
            env=self.env,
            level="meta",
            optimizer=params["optimizer"],
            device=self.device,
            checkpoint_dir=self.cp_dir,
            gamma=params["meta_gamma"],
            batch_size=params["meta_batch_size"],
            lr=params["lr"],
            target_net_update_freq=params["meta_target_update_rate"],
            tb_writer=self.tb_writer,
            use_per=params["meta_use_per"],
            restore_dir=params["meta_cp_dir"],
            restore_str=params["meta_cp_str"],
        )

    def _calc_episolon_vals(self, total_steps, level="atomic"):
        episolon_decay_steps = int(total_steps * self.hyperparams[f"{level}_eps_decay"])
        epsilon_decay_vals = np.linspace(
            start=self.hyperparams[f"{level}_eps_start"],
            stop=self.hyperparams[f"{level}_eps_end"],
            num=episolon_decay_steps
        )
        all_vals = np.append(
            epsilon_decay_vals,
            np.array(
                [self.hyperparams[f"{level}_eps_end"]] *
                int(
                    total_steps *
                    (1 - self.hyperparams[f"{level}_eps_decay"]) +
                    self.env.episode_step_limit[level]  # if we start new episode at total_steps - 1 ...
                )
            )
        )

        return all_vals

    def train(self, atomic_training_steps, meta_training_steps):
        atomic_step_counter = 0
        atomic_episode_counter = 0
        atomic_episode_rewards = []
        atomic_episode_component_rewards = None
        atomic_episode_successes = []
        atomic_best_mean = -10000
        atomic_epsilon_vals = self._calc_episolon_vals(atomic_training_steps + self.atomic_pretrain_steps)

        meta_step_counter = 0
        meta_episode_counter = 0
        meta_episode_rewards = []
        meta_episode_component_rewards = None
        meta_episode_successes = []
        meta_best_mean = -10000
        meta_epsilon_vals = self._calc_episolon_vals(meta_training_steps, "meta")

        # helpers for plotting
        decomposed_reward_plot_counter = 0

        # start tensorboard
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', self.cp_dir])
        url = tb.launch()

        print(f"Tensorboard running on {url}#scalars&_smoothingWeight=0.8")
        print()

        while meta_step_counter < meta_training_steps:
            meta_obs = self.env.meta_reset()
            meta_done = False
            meta_episode_success = False
            meta_episode_reward = np.zeros([self.env.n_reward_components["meta"]])
            meta_local_step_counter = 0  # so that we can still track meta steps even when atomic is pretraining

            while not meta_done:
                # select meta action...
                meta_action = self.meta_policy.select_action(meta_obs, epsilon=meta_epsilon_vals[meta_step_counter])
                self.tb_writer.add_scalar("agent/meta/epsilon", meta_epsilon_vals[meta_step_counter], meta_step_counter)
                self.env.do_meta_action(meta_action.cpu().detach().numpy().item())

                if atomic_episode_counter < atomic_training_steps:
                    obs = self.env.atomic_reset()
                    done = False
                    episode_success = False
                    episode_reward = np.zeros([self.env.n_reward_components["atomic"]])

                    while not done:
                        try:
                            atomic_episolon = atomic_epsilon_vals[atomic_step_counter]
                        except IndexError:  # happens because we keep training the meta controller when atomic steps are exceeded
                            atomic_episolon = atomic_epsilon_vals[-1]

                        action = self.atomic_policy.select_action(obs, epsilon=atomic_episolon)
                        self.tb_writer.add_scalar("agent/atomic/epsilon", atomic_episolon, atomic_step_counter)
                        new_obs, reward, done, msg_dict, episode_success = self.env.atomic_step(action.cpu().detach().numpy().item())
                        # new_obs, reward, done, msg_dict = self.env.step(0.7)
                        episode_reward += reward
                        atomic_step_counter += 1

                        # store transition in buffer (expand_dims so that we have shapes [1, n] instead of [n])
                        for idx, rc in enumerate(self.atomic_policy.reward_components):
                            if self.atomic_policy.use_per:
                                rc.replay_buffer.push(
                                    obs,
                                    action.item(),
                                    new_obs,
                                    reward[idx],
                                    not done  # invert because we use it as mask
                                )
                            else:
                                rc.replay_buffer.push(
                                    torch.tensor(np.expand_dims(obs, 0), dtype=torch.double, device=self.device),
                                    action.unsqueeze(0),  # insert dim so that it is of shape [1, 7] instead of [7]
                                    torch.tensor(np.expand_dims(new_obs, 0), dtype=torch.double, device=self.device),
                                    torch.tensor(np.expand_dims(reward[idx], 0), dtype=torch.double, device=self.device),  # reward for that component...
                                    torch.tensor(np.expand_dims(not done, 0), dtype=torch.double, device=self.device)
                                )

                        # move to new state
                        obs = new_obs

                        # (conditionally) optimize network and update target
                        self.atomic_policy.optimize_model()
                        self.atomic_policy.update_target_net()
                        self.atomic_policy.pred_eval_states(self.env.eval_states)

                    # atomic episode done
                    atomic_episode_rewards.append(sum(episode_reward))
                    if atomic_episode_component_rewards is None:
                        atomic_episode_component_rewards = episode_reward
                    else:
                        atomic_episode_component_rewards = np.vstack((atomic_episode_component_rewards, episode_reward))
                    self.tb_writer.add_scalar("reward/atomic/total", sum(episode_reward), atomic_episode_counter)
                    for c in range(len(self.atomic_policy.reward_components)):
                        self.tb_writer.add_scalar(f"reward/atomic/component{c}", episode_reward[c], atomic_episode_counter)
                    atomic_episode_successes.append(int(episode_success))
                    self.tb_writer.add_scalar("agent/atomic/success_rate", np.mean(atomic_episode_successes[-50:]), atomic_episode_counter)

                    running_mean = np.mean(atomic_episode_rewards[-50:])
                    if running_mean > atomic_best_mean:
                        print(f"E {atomic_episode_counter}: New best atomic mean, saving weights")
                        self.atomic_policy.save_components(atomic_episode_counter)
                        atomic_best_mean = running_mean

                    atomic_episode_counter += 1

                    if atomic_episode_counter % self.atomic_safe_freq == 0:
                        self.atomic_policy.save_components(atomic_episode_counter, mode="intermediate")

                # atomic episode handling done
                meta_local_step_counter += 1
                if meta_local_step_counter > self.env.episode_step_limit["meta"]:
                    meta_done = True

                if atomic_step_counter > self.atomic_pretrain_steps:
                    meta_step_counter += 1
                    meta_new_obs, meta_reward, meta_done, meta_episode_success, meta_info = self.env.eval_meta_action()
                    meta_episode_reward += meta_reward

                    # store transition in buffer (expand_dims so that we have shapes [1, n] instead of [n])
                    for idx, rc in enumerate(self.meta_policy.reward_components):
                         if self.meta_policy.use_per:
                             rc.replay_buffer.push(
                                 meta_obs,
                                 meta_action.item(),
                                 meta_new_obs,
                                 meta_reward[idx],
                                 not meta_done  # invert because we use it as mask
                             )
                         else:
                             rc.replay_buffer.push(
                                 torch.tensor(np.expand_dims(meta_obs, 0), dtype=torch.double, device=self.device),
                                 meta_action.unsqueeze(0),  # insert dim so that it is of shape [1, 7] instead of [7]
                                 torch.tensor(np.expand_dims(meta_new_obs, 0), dtype=torch.double, device=self.device),
                                 torch.tensor(np.expand_dims(meta_reward[idx], 0), dtype=torch.double, device=self.device),
                                 torch.tensor(np.expand_dims(not meta_done, 0), dtype=torch.double, device=self.device)
                         )

                    # move to new state
                    meta_obs = meta_new_obs  # doing this here is fine, because when we are still pretraining, meta_policy ist just taking random actions

                    # (conditionally) optimize network and update target
                    for _ in range(self.hyperparams["meta_policy_update_rate"]):
                        self.meta_policy.optimize_model()
                        self.meta_policy.update_target_net()

            # atomic episode done
            meta_episode_counter += 1
            meta_episode_rewards.append(sum(meta_episode_reward))
            if meta_episode_component_rewards is None:
                meta_episode_component_rewards = meta_episode_reward
            else:
                meta_episode_component_rewards = np.vstack((meta_episode_component_rewards, meta_episode_reward))
            self.tb_writer.add_scalar("reward/meta/total", sum(meta_episode_reward), meta_episode_counter)
            for c in range(len(self.meta_policy.reward_components)):
                self.tb_writer.add_scalar(f"reward/meta/component{c}", meta_episode_reward[c], meta_episode_counter)
            meta_episode_successes.append(int(meta_episode_success))
            self.tb_writer.add_scalar("agent/meta/success_rate", np.mean(meta_episode_successes[-50:]),
                                      meta_episode_counter)

            meta_running_mean = np.mean(meta_episode_rewards[-50:])
            if meta_running_mean > meta_best_mean:
                print(f"E {meta_episode_counter}: New best meta mean, saving weights")
                self.meta_policy.save_components(meta_episode_counter)
                meta_best_mean = meta_running_mean

            if meta_episode_counter % self.meta_safe_freq == 0:
                self.meta_policy.save_components(meta_episode_counter, mode="intermediate")

        print(f"\nTraining ended. Restart tensorboard with:\ntensorboard --logdir={self.cp_dir}\n")








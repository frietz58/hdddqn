import torch
from utils.replay_memory import ReplayMemory, Transition
from torch.autograd import Variable
from torch.nn import SmoothL1Loss
import os
import random
import numpy as np
from reward_component import RewardComponent


class Policy:
    def __init__(self,
                 model_class,
                 env,
                 level,
                 optimizer,
                 device,
                 checkpoint_dir,
                 batch_size,
                 gamma,
                 lr,
                 target_net_update_freq,
                 tb_writer,
                 tau=0.001,
                 replay_buffer_size=1000000,
                 restore_dir="",
                 restore_str="",
                 use_per=False
                 ):

        self.cp_dir = checkpoint_dir
        self.restore_dir = restore_dir
        self.restore_str = restore_str
        self.env = env
        self.level = level
        self.optimizer_fn = optimizer
        self.optimizer = None
        self.device = device
        self.batch_size = batch_size
        self.gamma = gamma
        self.loss_fn = SmoothL1Loss()
        self.lr = lr
        self.tau = tau
        self.hard_target_update_freq = target_net_update_freq
        self.torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tb_writer = tb_writer
        self.replay_buffer_size = replay_buffer_size
        self.use_per = use_per

        self.gradient_steps = 0

        # self.replay_buffer = ReplayMemory(
        #     replay_buffer_size,
        #     save_dir=checkpoint_dir,
        #     name="ReplayBuffer"
        # )

        self.reward_components = []
        self.setup_networks(model_class)

    def setup_networks(self, model_class):
        for c_idx in range(self.env.n_reward_components[self.level]):
            reward_component = RewardComponent(
                name=f"{self.level}_component{c_idx}",
                net=model_class,
                optimizer=self.optimizer_fn,
                lr=self.lr,
                n_obs=self.env.observation_space[f"{self.level}_observation_space"].shape[0],  # we want the only element in the tuple, not the tuple
                n_actions=self.env.action_space[f"{self.level}_action_space"].n,
                checkpoint_dir=self.cp_dir,
                cuda_device=self.torch_device,
                restore_dir=self.restore_dir,
                restore_str=self.restore_str,
                replay_buffer_size=self.replay_buffer_size,
                use_per=self.use_per
            )
            self.reward_components.append(reward_component)

    def select_action(self, obs, epsilon=None, return_components=False):
        sample = random.random()
        if sample > epsilon:
            if type(obs) != torch.Tensor:
                obs = torch.from_numpy(obs).to(self.device).double()

            stacked_pred = None
            for rc in self.reward_components:
                rc.policy_net.eval()
                pred = rc.predict(obs)
                if stacked_pred is None:
                    stacked_pred = pred
                else:
                    stacked_pred = torch.vstack((stacked_pred, pred))
                rc.policy_net.train()

            if len(self.reward_components) > 1:
                action = stacked_pred.sum(0).argmax(0).unsqueeze(0)  # we only want to sum along rows when we have multiple reward component rows...
            else:                                                    # unsqueeze so that it is row tensor
                action = stacked_pred.argmax().unsqueeze(0)
        else:
            action = torch.tensor([random.randrange(self.env.action_space[f"{self.level}_action_space"].n)], device=self.device).double()

        if return_components:
            # will break if return_components=True but we did exploratory action, but thats fine for now
            return action, stacked_pred
        else:
            return action

    def optimize_model(self):
        for rc_idx, rc in enumerate(self.reward_components):
            if len(rc.replay_buffer) < self.batch_size:
                return

            if self.use_per:
                state, action, reward, next_state, done, indices, weights = rc.replay_buffer.sample(self.batch_size)
                state_batch = torch.from_numpy(state).to(self.device)
                action_batch = torch.from_numpy(np.array(action)).unsqueeze(1).to(self.device)
                reward_batch = torch.from_numpy(np.array(reward)).unsqueeze(1).to(self.device)
                mask_batch = torch.from_numpy(np.array(done, dtype=np.int)).unsqueeze(1).to(self.device)
                next_state_batch = torch.from_numpy(np.array(next_state)).to(self.device)
                weights = torch.from_numpy(weights).unsqueeze(1).to(self.device)

            else:
                transitions = rc.replay_buffer.sample(self.batch_size)
                batch = Transition(*zip(*transitions))

                state_batch = torch.cat(batch.state)
                action_batch = torch.cat(batch.action)
                reward_batch = torch.cat(batch.reward).unsqueeze(1)
                mask_batch = torch.cat(batch.mask)
                mask_batch = mask_batch.unsqueeze(1)
                next_state_batch = torch.cat(batch.next_state)

                # TODO: init PER weights tensor with just ones

            with torch.no_grad():
                # get action from global policy for next state
                global_next_state_preds = torch.tensor(
                    np.zeros([self.batch_size, self.env.action_space[f"{self.level}_action_space"].n]), device=self.device).double()

                for _rc in self.reward_components:  # avoid overwriting variable...
                    local_next_state_preds = _rc.target_net(next_state_batch)
                    global_next_state_preds = global_next_state_preds + local_next_state_preds

                next_state_preds = global_next_state_preds
                global_next_state_actions = next_state_preds.max(dim=1).indices  # value of globally best action...
                global_next_state_actions = global_next_state_actions.unsqueeze(dim=1)  # make its shape [BATCH_SIZE, 1] instead of [BATCH_SIZE]

            # component_reward_batch = reward_batch[:, rc_idx].unsqueeze(1)  # select rewards for current component
            component_reward_batch = reward_batch

            local_next_state_preds = rc.target_net(next_state_batch)
            local_next_state_values = local_next_state_preds.gather(dim=1, index=global_next_state_actions)  # value corresponds to action taken by global policy
            expected_state_action_values = component_reward_batch + (self.gamma * mask_batch * local_next_state_values)

            state_preds = rc.policy_net(state_batch)
            state_action_values = state_preds.gather(dim=1, index=action_batch.long())

            # loss = self.loss_fn(state_action_values, expected_state_action_values)
            loss = (state_action_values - expected_state_action_values.detach()).pow(2) * weights
            loss = loss * weights
            prios = loss + 1e-5
            loss = loss.mean()

            self.tb_writer.add_scalar(f"training/{self.level}/component{rc_idx}/loss", loss, self.gradient_steps)

            rc.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rc.policy_net.parameters(), 1)
            rc.optimizer.step()

            # softly update target network with fraction of real update
            rc.soft_update(self.tau)

            if self.use_per:
                rc.replay_buffer.update_priorities(indices, prios.data.cpu().numpy())

        self.gradient_steps += 1

    def update_target_net(self):
        if self.gradient_steps % self.hard_target_update_freq == 0:
            for rc in self.reward_components:
                rc.target_net.load_state_dict(rc.policy_net.state_dict())

    def save_components(self, episode_counter, mode="best_mean"):
        if not os.path.exists(self.cp_dir):
            os.makedirs(self.cp_dir)

        for rc in self.reward_components:
            rc.save_model(episode_counter, mode=mode)

    def pred_eval_states(self, eval_states):
        for state_idx, eval_state in enumerate(eval_states):
            action, stacked_pred = self.select_action(np.array(eval_state), epsilon=0, return_components=True)
            for comp_idx in range(stacked_pred.shape[0]):
                best_value = stacked_pred[comp_idx, :].max(0)
                self.tb_writer.add_scalar(f"eval/comp{comp_idx}/state{state_idx}", best_value.values.item(), self.gradient_steps)






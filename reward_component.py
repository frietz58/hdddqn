import torch
from colorama import Fore, Style, Back
import numpy as np
import os
from torchinfo import summary
from utils.replay_memory import ReplayMemory, Transition
from utils.per import NaivePrioritizedBuffer


class RewardComponent:
    def __init__(
            self,
            name: str,
            net: torch.nn.Module,
            optimizer: torch.optim,
            lr: float,
            n_obs: int,
            n_actions: int,
            checkpoint_dir: str,
            cuda_device=torch.device,
            restore_dir="",
            restore_str="",
            replay_buffer_size=1000000,
            use_per=False
    ):
        """
        This is one a class for one reward component approximator.
        :param name: The str name for this component
        :param net: The torch network class to use for prediction
        :param optimizer: The optimizer to use for the net
        :param lr: The learning rate
        :param n_obs: The length of the observation vector (N scalars in it)
        :param n_actions: The length of the discrete action vector
        :param checkpoint_dir: The directory to save checkpoints and other files
        :param restore_cp: A .pth torch checkpoint file to load network weights from
        """
        self.name = name
        self.policy_net = None
        self.target_net = None
        self.optimizer = None
        self.lr = lr
        self.n_obs = n_obs
        self.n_actions = n_actions
        self.checkpoint_dir = checkpoint_dir
        self.restore_dir = restore_dir
        self.restore_str = restore_str
        self.cuda_device = cuda_device
        self.use_per = use_per

        if use_per:
            self.replay_buffer = NaivePrioritizedBuffer(replay_buffer_size)
        else:
            self.replay_buffer = ReplayMemory(
                replay_buffer_size,
                save_dir=checkpoint_dir,
                name="ReplayBuffer"
            )

        self.interactions_done = 0  # number of predictions done, used in model optimization

        self._setup_nets(model=net, optim=optimizer, lr=lr)

    def _setup_nets(self, model, optim, lr):
        """
        Simply sets up the torch neural models to used by the policy.
        :param model: The torch network to use
        :param optim: The torch optimizer to use
        :param lr: The learning rate for the optimizer
        """
        self.policy_net = model(
            state_size=self.n_obs,
            action_size=self.n_actions,
            input_dtype=torch.double
        )
        if self.restore_dir:
            files_in_restore_dir = os.listdir(self.restore_dir)
            restored = False
            for file in files_in_restore_dir:
                if ".pth" in file and f"{self.name}" in file and self.restore_str in file:
                    self.policy_net.load_state_dict(torch.load(os.path.join(self.restore_dir, file)))
                    print(f"Restored component {self.name} from CP file {file}")
                    restored = True
                    break

            if not restored:
                print(Back.RED + "Given restore DIR but didn't fint CP files matching component!" + Style.RESET_ALL)

        self.optimizer = optim(self.policy_net.parameters(), lr=lr)

        self.target_net = model(
            state_size=self.n_obs,
            action_size=self.n_actions,
            input_dtype=torch.double
        )

        self.policy_net.double()
        self.target_net.double()

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

    def predict(self, state, use_target_net=False):
        """
        Predict action scores for state
        :param state: The state for which we want the prediction
        :param use_target_net: Get prediction from target net
        :return: The network action for the state input
        """
        if type(state) != torch.Tensor:
            state = torch.from_numpy(state).to(self.cuda_device)

        self.interactions_done += 1

        if use_target_net:
            return self.target_net(state)
        else:
            return self.policy_net(state)

    def save_model(self, episode_counter, mode="avg_best"):
        """
        Saves the policy net to disk. Checks for already existing saves of this kind and overrides them if found.
        :param episode_counter: Episode counter, int
        :param mode: Save 'mode',
            + 'avg': Save if this is avg best model from past 100 episodes.
            + 'latest': Save the newest version of the model.
        """
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        # delete existing checkpoint of this kind (we have a newer one with a different name now...)
        for f in os.listdir(self.checkpoint_dir):
            if mode != "intermediate":
                if mode in f and self.name in f and ".pth" in f:
                    os.remove(os.path.join(self.checkpoint_dir, f))

        torch.save(self.policy_net.state_dict(), os.path.join(
            self.checkpoint_dir,
            "{}_{}_model_e{}.pth".format(mode, self.name, episode_counter)
        ))

    def soft_update(self, tau):
        # softly update target network params
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)




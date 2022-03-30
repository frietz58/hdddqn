from collections import namedtuple
import random
import numpy as np
import os

import torch

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'mask'))


class ReplayMemory(object):
    def __init__(self, capacity, save_dir=".", name="replay_buffer"):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.name = name
        self.save_dir = os.path.join(save_dir, name)
        self.push_counter = 0

    def _save_to_disk(self, transition):
        num = "{:07}".format(self.push_counter)
        file_name = num + "_" + self.name

        if type(transition.action) == torch.Tensor:
            action = transition.action.detach().cpu()
        else:
            action = transition.action
        if type(transition.reward) == torch.Tensor:
            reward = transition.reward.detach().cpu()
        else:
            reward = transition.reward

        np.savez(
            os.path.join(self.save_dir, file_name),
            state=transition.state,
            action=action,
            next_state=transition.next_state,
            reward=reward
        )
        
    def _load_from_disk(self, dir):
        print(dir)
        files = os.listdir(dir)
        files.sort()

        for file in files:
            if ".npz" in file:
                file_path = os.path.join(dir, file)
                data = np.load(file_path, allow_pickle=True)

                state = data["state"]
                action = data["action"]
                next_state = data["next_state"]
                reward = data["reward"]

                self.push(state, action, next_state, reward)

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        t = Transition(*args)
        self.memory[self.position] = t
        self.position = (self.position + 1) % self.capacity
        # self._save_to_disk(t)
        self.push_counter += 1

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    

if __name__ == "__main__":
    mem = ReplayMemory(10000)
    mem._load_from_disk(
        "/home/finn/msc_thesis/DQN/hdr_dqn_coppelia_fullTask_fullDecomposed_doubleQ/checkpoints/2021.10.29-16:55:27default/0/replay_buffer_AtomicPolicy"
    )
    print(len(mem.memory))

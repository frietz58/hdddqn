
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """ Actor (Policy) Model."""

    def __init__(self, state_size, action_size, fc1_unit=64,
                 fc2_unit=64, input_dtype=torch.double):
        """
        Initialize parameters and build model.
        Params
        =======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            fc1_unit (int): Number of nodes in first hidden layer
            fc2_unit (int): Number of nodes in second hidden layer
        """
        self.input_dtype = input_dtype
        self.torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super(QNetwork, self).__init__()  # calls __init__ method of nn.Module class
        self.fc1 = nn.Linear(state_size, fc1_unit, device=self.torch_device, dtype=self.input_dtype)
        self.fc2 = nn.Linear(fc1_unit, fc2_unit, device=self.torch_device, dtype=self.input_dtype)
        self.fc3 = nn.Linear(fc2_unit, action_size, device=self.torch_device, dtype=self.input_dtype)

    def forward(self, x):
        """
        Build a network that maps state -> action values.
        x = state
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

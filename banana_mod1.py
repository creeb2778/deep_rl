import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        #import the __init__ from nn.Module
        super(QNetwork, self).__init__()
        
        self.seed = torch.manual_seed(seed)
        self.fc1 = torch.nn.Linear(state_size, 150)
        self.fc2 = torch.nn.Linear(150, 150)
        self.fc3 = torch.nn.Linear(150, 60)
        self.fc4 = torch.nn.Linear(60, 60)
        self.fc5 = torch.nn.Linear(60, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        
       
        x = F.relu(self.fc1(state))
        
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return self.fc5(x)
        
        
        
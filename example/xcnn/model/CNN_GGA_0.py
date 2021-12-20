import torch.nn as nn
import torch.nn.functional as F


class CNN_GGA_0(nn.Module):
    def __init__(self):
        super(CNN_GGA_0, self).__init__()
        self.rho_type = "GGA"
        self.conv1 = nn.Conv3d(4, 8, 4) # 9x9x9 -> 6x6x6
        self.fc1 = nn.Linear(216, 108)
        self.fc2 = nn.Linear(108, 50)
        self.fc3 = nn.Linear(50, 25)
        self.fc4 = nn.Linear(25, 1)

    def forward(self, x):
        # x shape: 4 x 9 x 9 x 9
        # for GGA-like NN, use electron density and its gradients

        x = F.max_pool3d(F.elu(self.conv1(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = F.elu(self.fc3(x))
        x = self.fc4(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    

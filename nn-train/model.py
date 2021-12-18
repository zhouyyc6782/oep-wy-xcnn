import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_GGA_1(nn.Module):
    def __init__(self):
        super(CNN_GGA_1, self).__init__()
        self.rho_type = "GGA"
        self.conv1 = nn.Conv3d(4, 8, 4) # 4@9x9x9 -> 8@6x6x6, 4x4x4 kernel
        self.conv2 = nn.Conv3d(8, 16, 3) # 8@6x6x6 -> 16@4x4x4, 3x3x3 kernel
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)

    def forward(self, x):
        # x shape: 4 x 9 x 9 x 9
        # for GGA-like NN, use electron density and its gradients

        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.max_pool3d(x, 2)
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


class CNN_GGA_1_zsym(nn.Module):
    def __init__(self):
        super(CNN_GGA_1_zsym, self).__init__()
        self.rho_type = "GGA"

        self.conv1 = nn.Conv3d(4, 8, 4) # 4@9x9x9 -> 8@6x6x6, 4x4x4 kernel
        self.conv2 = nn.Conv3d(8, 16, 3) # 8@6x6x6 -> 16@4x4x4, 3x3x3 kernel
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)

    def forward(self, x):
        # x shape: batch_size x 2 x 4 x 9 x 9 x 9

        xp = x[:, 0]
        xp = F.elu(self.conv1(xp))
        xp = F.elu(self.conv2(xp))
        xp = F.max_pool3d(xp, 2)
        xp = xp.view(-1, self.num_flat_features(xp))
        xp = F.elu(self.fc1(xp))
        xp = F.elu(self.fc2(xp))
        xp = F.elu(self.fc3(xp))
        xp = self.fc4(xp)

        xm = x[:, 1]
        xm = F.elu(self.conv1(xm))
        xm = F.elu(self.conv2(xm))
        xm = F.max_pool3d(xm, 2)
        xm = xm.view(-1, self.num_flat_features(xm))
        xm = F.elu(self.fc1(xm))
        xm = F.elu(self.fc2(xm))
        xm = F.elu(self.fc3(xm))
        xm = self.fc4(xm)

        return (xm, xp)

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


import torch.nn as nn
import torch.nn.functional as F


class CNN_LDA_0(nn.Module):
    def __init__(self):
        super(CNN_LDA_0, self).__init__()
        self.rho_type = "LDA"
        self.conv1 = nn.Conv3d(1, 6, 4) # 9x9x9 -> 6x6x6
        self.fc1 = nn.Linear(162, 81)
        self.fc2 = nn.Linear(81, 40)
        self.fc3 = nn.Linear(40, 1)

    def forward(self, x):
        # x shape: 4 x 9 x 9 x 9
        # for LDA-like NN, use only electron density, 
        # i.e. [[0][:, :, :]]

        # extract first channel and add back one dimension
        # 4 x 9 x 9 x 9 -> 9 x 9 x 9 -> 1 x 9 x 9 x 9
        x.data = x.data[:, 0].unsqueeze_(1) 
        x = F.max_pool3d(F.elu(self.conv1(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


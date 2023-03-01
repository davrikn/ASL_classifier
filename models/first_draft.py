import torch
import torch.nn as nn


class FirstDraftCNN(nn.Module):
    
    def __init__(self):
        super(FirstDraftCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)   # 6 channels out, kernel size 7
        self.pool = nn.MaxPool2d(2, 2)    # 2x2 maxpool (strife). Image now has shape 49x49
        self.conv2 = nn.Conv2d(6, 16, 5)  # 6 channel input, 16 channel output. Image now has dimension 47x47
        self.fc1 = nn.Linear(16*47*47, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 29)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
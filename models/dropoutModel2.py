import torch
import torch.nn as nn
import numpy as np

class DropoutModel(nn.Module):
    name = "model_v2"
    final_conv_output = 32

    def __init__(self):
        super(DropoutModel, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, self.final_conv_output, kernel_size=5, stride=1, padding=2)
        self.fc1 = nn.Linear(self.final_conv_output * 24 * 24, 512) # 192/2^-3 = 24
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, 29)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = torch.relu(self.conv4(x))
        x = x.view(-1, self.final_conv_output * 24 * 24)
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        return x

    def predict(self, x: list[list[list[int]]]):
        """
        :param x: A 3-dimensional array with dimensions [RGB, x, y]
        :return: class number which is predicted by the model
        """
        outputs = self(x)
        _, predicted = torch.max(outputs, 1)
        return np.argmax(predicted.cpu().detach().numpy())

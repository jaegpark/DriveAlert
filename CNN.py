import torch
from torch import nn
import torch.nn.functional as F


# define CNN Architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # conv layer
        # sees 24x24 x3 (RGB)
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1) # in depth = 3, out depth = 16, ksize = 3, padding 1, stride 1 (default)
        
        # sees 12x12 x16
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1) # in depth = 16, out depth = 32, ksize = 3, padding 1, stride 1

        # sees 6x6 x32
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        
        # final out of conv: 3x3 x64

        self.fc1 = nn.Linear(3*3*64, 100, bias=True)
        self.fc2 = nn.Linear(100, 2, bias=True)
        self.dropout = nn.Dropout(p=0.25)

        self.pool = nn.MaxPool2d(2, 2)
        
    def forward(self, x):
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.view(-1, 3*3*64)

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)

        return F.log_softmax(x)

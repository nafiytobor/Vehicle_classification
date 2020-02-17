import torch
import torch.nn as nn

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=6,kernel_size=5,stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.fc1=nn.Sequential(
            nn.Linear(in_features=61*61*16,out_features=500),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=500, out_features=196),
            nn.Sigmoid()
        )

    def forward(self, input):
        conv1_output=self.conv1(input)#[256,256,3]--->[252,252,6]--->[126,126,6]
        conv2_output = self.conv2(conv1_output)  # [126,126,6]--->[122,122,16]--->[61,61,16]
        conv2_output=conv2_output.view(-1,61*61*16)#将[n,4,4,16]维度转化为[n,4*4*16]
        fc1_output=self.fc1(conv2_output)#[n,256]--->[n,120]
        fc2_output=self.fc2(fc1_output)#[n,120]-->[n,84]
        return fc2_output



import torch
from torch.nn import Conv2d, ReLU, Softmax

class EIIE(torch.nn):

    # initializer
    def __init__(self):
        self.conv2d_1 = Conv2d(in_channels=3, out_channels=2, kernel_size=(11,48))
        self.relu1 = ReLU()
        self.conv2d_2 = Conv2d(in_channels=2, out_channels=21, kernel_size=(11,1))
        self.relu2 = ReLU()
        self.conv2d_3 = Conv2d(in_channels=21, out_channels=1, kernel_size=(11,1))
        self.softmax = Softmax()

    # compute the forward pass
    def forward(self, x):
        x = self.conv2d_1(x)
        x = self.relu1(x)
        x = self.conv2d_2(x)
        x = self.relu2(x)
        x = self.conv2d_3(x)
        output = self.softmax(x)
        return output


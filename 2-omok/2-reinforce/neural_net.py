from torch import nn
from torch.nn import init


class PVNet(nn.Module):
    def __init__(self, in_channel, state_size):
        super(PVNet, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.conv2d = nn.Conv2d(64, 1, kernel_size=1, bias=False)
        self.policy = nn.Sequential(
            nn.Linear(state_size**2, state_size**2),
            nn.Softmax(dim=1)
        )

        self.value = nn.Sequential(
            nn.Linear(state_size**2, 1),
            nn.Tanh()
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_uniform(m.weight, gain=0.01)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.feature(x)
        x = self.conv2d(x)
        x = x.view(x.size(0), -1)
        p = self.policy(x)
        v = self.value(x)
        return p, v


if __name__ == '__main__':
    # test
    import torch
    from torch.autograd import Variable
    import numpy as np
    use_cuda = torch.cuda.is_available()
    Tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
    model = PVNet(in_channel=5, state_size=9)
    state = np.ones((5, 9, 9))
    state_input = Variable(torch.FloatTensor([state]))
    p, v = model(state_input)
    print('cuda:', use_cuda)
    print('P: {}\nV: {}'.format(p, v))

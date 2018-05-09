from torch import nn
from torch.nn import init
import torch.nn.functional as F


def conv3x3(in_planes, out_planes):

    return nn.Conv2d(in_planes, out_planes,
                     kernel_size=3,
                     padding=1,
                     bias=False)


class ResBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = F.relu(out)

        return out


class PolicyHead(nn.Module):
    def __init__(self, planes, board_size):
        super(PolicyHead, self).__init__()
        self.policy_head = nn.Conv2d(planes, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size**2, board_size**2)

    def forward(self, x):
        out = self.policy_head(x)
        out = self.policy_bn(out)
        out = F.relu(out)
        out = out.view(out.size(0), -1)
        out = self.policy_fc(out)
        out = F.log_softmax(out, dim=1)
        out = out.exp()

        return out


class ValueHead(nn.Module):
    def __init__(self, planes, board_size):
        super(ValueHead, self).__init__()
        self.value_head = nn.Conv2d(planes, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size**2, planes)
        self.value_fc2 = nn.Linear(planes, 1)

    def forward(self, x):
        out = self.value_head(x)
        out = self.value_bn(out)
        out = F.relu(out)
        out = out.view(out.size(0), -1)
        out = self.value_fc1(out)
        out = F.relu(out)
        out = self.value_fc2(out)
        out = F.tanh(out)
        out = out.view(out.size(0))

        return out


class PVNet(nn.Module):
    def __init__(self, n_block, inplanes, planes, board_size):
        super(PVNet, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.layers = self._make_layer(ResBlock, planes, n_block)
        self.policy_head = PolicyHead(planes, board_size)
        self.value_head = ValueHead(planes, board_size)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_uniform(m.weight)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, n_block):
        blocks = []
        for i in range(n_block):
            blocks.append(block(planes, planes))

        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.layers(x)
        p = self.policy_head(x)
        v = self.value_head(x)

        return p, v


if __name__ == '__main__':
    # test
    import torch
    from torch.autograd import Variable
    import numpy as np
    use_cuda = torch.cuda.is_available()
    Tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
    model = PVNet(20, 17, 128, 3)
    state = np.ones((17, 3, 3))
    state_input = Variable(torch.FloatTensor([state]))
    p, v = model(state_input)
    print('cuda:', use_cuda)
    print('P: {}\nV: {}'.format(p, v))

import numpy as np
import torch
from torch import nn
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
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class PolicyHead(nn.Module):
    def __init__(self, planes, board_size):
        super(PolicyHead, self).__init__()
        self.policy_head = nn.Conv2d(planes, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.relu = nn.ReLU()
        self.policy_fc = nn.Linear(board_size**2 * 2, board_size**2)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        out = self.policy_head(x)
        out = self.policy_bn(out)
        out = self.relu(out)
        out = out.view(out.size(0), -1)
        out = self.policy_fc(out)
        out = self.log_softmax(out)
        out = out.exp()
        return out


class ValueHead(nn.Module):
    def __init__(self, planes, board_size):
        super(ValueHead, self).__init__()
        self.value_head = nn.Conv2d(planes, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.relu = nn.ReLU()
        self.value_fc1 = nn.Linear(board_size**2, planes)
        self.value_fc2 = nn.Linear(planes, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = self.value_head(x)
        out = self.value_bn(out)
        out = self.relu(out)
        out = out.view(out.size(0), -1)
        out = self.value_fc1(out)
        out = self.relu(out)
        out = self.value_fc2(out)
        out = self.tanh(out)
        out = out.view(out.size(0))
        return out


class PVNet(nn.Module):
    def __init__(self, n_block, inplanes, planes, board_size):
        super(PVNet, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.layers = self._make_layer(ResBlock, planes, n_block)
        self.policy_head = PolicyHead(planes, board_size)
        self.value_head = ValueHead(planes, board_size)

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, n_block):
        blocks = []
        for i in range(n_block):
            blocks.append(block(planes, planes))
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layers(x)
        p = self.policy_head(x)
        v = self.value_head(x)
        return p, v


class NoisyLinear(nn.Linear):
    def __init__(self, inplanes, planes, sigma_zero=0.25, bias=False):
        super(NoisyLinear, self).__init__(inplanes, planes, bias=bias)
        sigma_init = sigma_zero / np.math.sqrt(inplanes)
        self.sigma_weight = nn.Parameter(
            torch.full((planes, inplanes), sigma_init))
        self.register_buffer('epsilon_input', torch.zeros(1, inplanes))
        self.register_buffer('epsilon_output', torch.zeros(planes, 1))
        if bias:
            self.sigma_bias = nn.Parameter(
                torch.full((planes,), sigma_init))

    def forward(self, x):
        self.epsilon_input.normal_()
        self.epsilon_output.normal_()

        def eps_generator(arg):
            return torch.sign(arg) * torch.sqrt(torch.abs(arg))

        eps_in = eps_generator(self.epsilon_input.data)
        eps_out = eps_generator(self.epsilon_output.data)
        bias = self.bias
        if bias is not None:
            bias = bias + self.sigma_bias * eps_out.t()
        noise_v = torch.mul(eps_in, eps_out)
        return F.linear(x, self.weight + self.sigma_weight * noise_v, bias)


class NoisyPolicyHead(nn.Module):
    def __init__(self, planes, board_size, sigma_zero):
        super(NoisyPolicyHead, self).__init__()
        self.policy_head = nn.Conv2d(planes, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.relu = nn.ReLU()
        self.policy_fc1 = NoisyLinear(board_size**2 * 2,
                                      board_size**2 * 2,
                                      sigma_zero)
        self.policy_fc2 = NoisyLinear(board_size**2 * 2,
                                      board_size**2,
                                      sigma_zero)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        out = self.policy_head(x)
        out = self.policy_bn(out)
        out = self.relu(out)
        out = out.view(out.size(0), -1)
        out = self.policy_fc1(out)
        out = self.relu(out)
        out = self.policy_fc2(out)
        out = self.log_softmax(out)
        out = out.exp()
        
        return out


class NoisyPVNet(nn.Module):
    def __init__(self, n_block, inplanes, planes, board_size, sigma_zero):
        super(NoisyPVNet, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.layers = self._make_layer(ResBlock, planes, n_block)
        self.policy_head = NoisyPolicyHead(planes, board_size, sigma_zero)
        self.value_head = ValueHead(planes, board_size)

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, n_block):
        blocks = []
        for i in range(n_block):
            blocks.append(block(planes, planes))
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layers(x)
        p = self.policy_head(x)
        v = self.value_head(x)
        return p, v


class PVNetW(nn.Module):
    def __init__(self, in_channel, state_size):
        super(PVNetW, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(in_channel, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.conv2d = nn.Conv2d(128, 1, kernel_size=1, bias=False)

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
                nn.init.kaiming_normal_(m.weight.data)
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
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
    from torch.autograd import Variable
    use_cuda = torch.cuda.is_available()
    Tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
    model = PVNet(20, 5, 64, 9)
    state = np.ones((5, 9, 9))
    state_input = Variable(torch.FloatTensor([state]))
    p, v = model(state_input)
    print('cuda:', use_cuda)
    print('P: {}\nV: {}'.format(p, v))

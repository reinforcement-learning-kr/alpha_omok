import numpy as np
import torch
from torch import nn


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
        self.relu = nn.ReLU(True)
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
        self.relu = nn.ReLU(True)
        self.policy_fc = nn.Linear(board_size**2 * 2, board_size**2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        out = self.policy_head(x)
        out = self.policy_bn(out)
        out = self.relu(out)
        out = out.view(out.size(0), -1)
        out = self.policy_fc(out)
        out = self.softmax(out)
        return out


class ValueHead(nn.Module):
    def __init__(self, planes, board_size):
        super(ValueHead, self).__init__()
        self.value_head = nn.Conv2d(planes, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.relu = nn.ReLU(True)
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
        self.relu = nn.ReLU(True)
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

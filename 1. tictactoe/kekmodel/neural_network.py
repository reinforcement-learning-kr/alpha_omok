# -*- coding: utf-8 -*-
import torch
from torch.autograd import Variable
import torch.nn as nn

import numpy as np


# 신경망 클래스: forward 자동 호출
class PolicyValueNet(nn.Module):
    def __init__(self, num_channel):
        super(PolicyValueNet, self).__init__()
        # convolutional layer
        self.conv = nn.Conv2d(9, num_channel, kernel_size=3, padding=1)
        self.conv_bn = nn.BatchNorm2d(num_channel)
        self.conv_relu = nn.ReLU(inplace=True)

        # residual layer 1
        self.conv1 = nn.Conv2d(num_channel, num_channel,
                               kernel_size=3, padding=1)
        self.conv1_bn = nn.BatchNorm2d(num_channel)
        self.conv1_relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_channel, num_channel,
                               kernel_size=3, padding=1)
        self.conv2_bn = nn.BatchNorm2d(num_channel)
        # forward엔 여기에 skip connection 추가 필요
        self.conv2_relu = nn.ReLU(inplace=True)

        # residual layer 2
        self.conv3 = nn.Conv2d(num_channel, num_channel,
                               kernel_size=3, padding=1)
        self.conv3_bn = nn.BatchNorm2d(num_channel)
        self.conv3_relu = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(num_channel, num_channel,
                               kernel_size=3, padding=1)
        self.conv4_bn = nn.BatchNorm2d(num_channel)
        # forward엔 여기에 skip connection 추가 필요
        self.conv4_relu = nn.ReLU(inplace=True)

        # residual layer 3
        self.conv5 = nn.Conv2d(num_channel, num_channel,
                               kernel_size=3, padding=1)
        self.conv5_bn = nn.BatchNorm2d(num_channel)
        self.conv5_relu = nn.ReLU(inplace=True)
        self.conv6 = nn.Conv2d(num_channel, num_channel,
                               kernel_size=3, padding=1)
        self.conv6_bn = nn.BatchNorm2d(num_channel)
        # forward엔 여기에 skip connection 추가 필요
        self.conv6_relu = nn.ReLU(inplace=True)

        # residual layer 4
        self.conv7 = nn.Conv2d(num_channel, num_channel,
                               kernel_size=3, padding=1)
        self.conv7_bn = nn.BatchNorm2d(num_channel)
        self.conv7_relu = nn.ReLU(inplace=True)
        self.conv8 = nn.Conv2d(num_channel, num_channel,
                               kernel_size=3, padding=1)
        self.conv8_bn = nn.BatchNorm2d(num_channel)
        # forward엔 여기에 skip connection 추가 필요
        self.conv8_relu = nn.ReLU(inplace=True)

        # residual layer 5
        self.conv9 = nn.Conv2d(num_channel, num_channel,
                               kernel_size=3, padding=1)
        self.conv9_bn = nn.BatchNorm2d(num_channel)
        self.conv9_relu = nn.ReLU(inplace=True)
        self.conv10 = nn.Conv2d(num_channel, num_channel,
                                kernel_size=3, padding=1)
        self.conv10_bn = nn.BatchNorm2d(num_channel)
        # forward엔 여기에 skip connection 추가 필요
        self.conv10_relu = nn.ReLU(inplace=True)

        # 정책 헤드: 정책함수 인풋 받는 곳
        self.policy_head = nn.Conv2d(num_channel, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_relu = nn.ReLU(inplace=True)
        self.policy_fc = nn.Linear(18, 9)
        self.policy_softmax = nn.Softmax(dim=1)

        # 가치 헤드: 가치함수 인풋 받는 곳
        self.value_head = nn.Conv2d(num_channel, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_relu1 = nn.ReLU(inplace=True)
        self.value_fc = nn.Linear(9, num_channel)
        self.value_relu2 = nn.ReLU(inplace=True)
        self.value_scalar = nn.Linear(num_channel, 1)
        self.value_out = nn.Tanh()

        # weight 초기화 (xavier)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, state):
        # convolutional layer
        x = self.conv(state)
        x = self.conv_bn(x)
        x = self.conv_relu(x)

        # residual layer 1
        residual = x
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = self.conv1_relu(x)
        x = self.conv2(x)
        x = self.conv2_bn(x)
        x += residual  # skip connection
        x = self.conv2_relu(x)

        # residual layer 2
        residual = x
        x = self.conv3(x)
        x = self.conv3_bn(x)
        x = self.conv3_relu(x)
        x = self.conv4(x)
        x = self.conv4_bn(x)
        x += residual  # skip connection
        x = self.conv4_relu(x)

        # residual layer 3
        residual = x
        x = self.conv5(x)
        x = self.conv5_bn(x)
        x = self.conv5_relu(x)
        x = self.conv6(x)
        x = self.conv6_bn(x)
        x += residual  # skip connection
        x = self.conv6_relu(x)

        # residual layer 4
        residual = x
        x = self.conv7(x)
        x = self.conv7_bn(x)
        x = self.conv7_relu(x)
        x = self.conv8(x)
        x = self.conv8_bn(x)
        x += residual  # skip connection
        x = self.conv8_relu(x)

        # residual layer 5
        residual = x
        x = self.conv9(x)
        x = self.conv9_bn(x)
        x = self.conv9_relu(x)
        x = self.conv10(x)
        x = self.conv10_bn(x)
        x += residual  # skip connection
        x = self.conv10_relu(x)

        # policy head
        p = self.policy_head(x)
        p = self.policy_bn(p)
        p = self.policy_relu(p)
        p = p.view(p.size(0), -1)  # 펼치기
        p = self.policy_fc(p)
        p = self.policy_softmax(p)

        # value head
        v = self.value_head(x)
        v = self.value_bn(v)
        v = self.value_relu1(v)
        v = v.view(v.size(0), -1)
        v = self.value_fc(v)
        v = self.value_relu2(v)
        v = self.value_scalar(v)
        v = self.value_out(v)

        return p, v


# test
if __name__ == "__main__":
    net = PolicyValueNet(128)
    print(net)
    x = Variable(torch.from_numpy(
        np.zeros((9, 3, 3), 'float')).float().unsqueeze(0))
    p, v = net(x)
    print(v.data.numpy()[0])

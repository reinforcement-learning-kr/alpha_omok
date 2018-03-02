# -*- coding: utf-8 -*-
import neural_net_5block

import time
import pickle

import slackweb

import torch
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils import data

start = time.time()

# Hyper Parameters
EPOCHS = 64
BATCH_SIZE = 32
LR = 0.2
L2 = 0.0001
CHANNEL = 128

# data load
with open('data/train_dataset_s800_g800.pickle', 'rb') as f:
    dataset = pickle.load(f)
train_dataset = data.DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# 신경망 생성 및 최적화 인스턴스 생성
pv_net = neural_net_5block.PolicyValueNet(CHANNEL).cuda()
optimizer = torch.optim.SGD(pv_net.parameters(), lr=LR, momentum=0.9, weight_decay=L2)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', min_lr=2e-4, patience=5, verbose=1)

# print spec
spec = {'epoch': EPOCHS, 'batch size': BATCH_SIZE, 'optim': 'SGD', **optimizer.defaults}
print(spec)

# train
step = 0
for epoch in range(EPOCHS):
    val_loss = 0
    for i, (s, pi, z) in enumerate(train_dataset):
        s = Variable(s.view(BATCH_SIZE, 9, 3, 3).float(), requires_grad=True).cuda()
        pi = Variable(pi.view(1, BATCH_SIZE * 9).float(), requires_grad=False).cuda()
        z = Variable(z.float(), requires_grad=False).cuda()

        # forward and backward
        optimizer.zero_grad()
        p, v = pv_net(s)
        p = p.view(BATCH_SIZE * 9, 1)
        loss = ((z - v).pow(2).sum() - torch.matmul(pi, torch.log(p))) / BATCH_SIZE
        loss.backward()
        optimizer.step()
        step += 1
        val_loss += loss.data[0]

        # step check
        if (i + 1) % 32 == 0:
            print('Epoch [{:d}/{:d}]  Loss: [{:0.4f}]  Step: [{:d}/{:d}]'.format(
                epoch + 1,
                EPOCHS, val_loss[0] / (i + 1),
                (i + 1) * BATCH_SIZE, len(train_dataset) * BATCH_SIZE))

    # epoch check
    finish = round(float(time.time() - start))
    print('Finished {} Epoch in {}s'.format(epoch + 1, finish))
    scheduler.step(val_loss[0], epoch)

# Save the Model
torch.save(pv_net.state_dict(), 'data/model_step{}.pickle'.format(step * BATCH_SIZE))

# 메시지 보내기
finish = round(float(time.time() - start))
slack = slackweb.Slack(
    url="https://hooks.slack.com/services/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")
slack.notify(text='Finished {} Epoch in {}s [UBT]'.format(epoch + 1, finish))

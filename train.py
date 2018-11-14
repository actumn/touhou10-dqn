import math
import random

import matplotlib.pyplot as plt
import numpy as np
from collections import deque, namedtuple
from skimage import color, transform, exposure

from th10.game import TH10
from th10.process import find_process, image_grab
from th10.memory_reader import MemoryReader
from config import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
game = TH10()
memory = deque()


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(img_channels, 32, kernel_size=8, stride=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=8, stride=2)
        self.pool1 = nn.MaxPool2d(kernel_size=8, stride=1, padding=2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=4, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=4, stride=1, padding=1)

        self.conv5 = nn.Conv2d(64, 128, kernel_size=2, stride=1)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=2, stride=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)

        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.head = nn.Linear(128, num_of_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)

        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool3(x)

        x = x.view(x.size(0), -1)  # <-- flatten
        x = F.dropout(F.relu(self.fc1(x)))
        x = F.relu(self.fc2(x))
        x = F.dropout(F.relu(self.fc3(x)))
        x = self.head(x)
        return x


def transform_state(single_state):
    single_state = transform.resize(single_state, (transform_height, transform_width))
    # numpy HWC to torch CHW
    single_state = single_state.transpose((2, 0, 1))
    single_state = torch.from_numpy(single_state)
    single_state = single_state.unsqueeze(0)
    single_state = single_state.to(device, dtype=torch.float)
    return single_state


model = DQN().to(device)
model.load_state_dict(torch.load('./weights'))
optimizer = optim.Adam(model.parameters())
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


state, _, _ = game.play(-1)
state = transform_state(state)
state = torch.cat((state, state, state, state), 1)

steps = 0
while True:
    loss = 0
    Q_sa = [0]

    eps_threshold = eps_end + (eps_start - eps_end) * math.exp(-1. * steps / eps_decay)
    if random.random() <= eps_threshold:
        action_index = np.random.randint(num_of_actions)  # choose a random action
    else:
        q_index = model(state).max(1)[1]  # input a stack of 4 images, get the prediction
        action_index = q_index.item()

    next_state, reward, is_terminal = game.play(action_index)
    next_state = transform_state(next_state)
    next_state = torch.cat((next_state, state[:, :9]), 1)

    '''
    We need enough states in our experience replay deque so that we can take a random sample from it of the size we declared.
    Therefore we wait until a certain number and observe the environment until we're ready.
    '''
    memory.append((state, action_index, next_state, reward))
    if len(memory) > exp_replay_memory:
        memory.popleft()

    if len(memory) > batch_size:
        transitions = random.sample(memory, batch_size)
        batch = Transition(*zip(*transitions))

        inputs = torch.cat(batch.state)
        targets = model(inputs)

        input_next_state = torch.cat(batch.next_state)
        Q_sa = model(input_next_state).max(1)[0]

        train_reward = torch.tensor(batch.reward, device=device, dtype=torch.float)
        targets[torch.arange(batch_size), batch.action] = train_reward + gamma * Q_sa

        outputs = model(inputs)
        loss = F.mse_loss(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        for param in model.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()

    state = next_state
    steps += 1

    if steps % timesteps_to_save_weights == 0:
        torch.save(model.state_dict(), './weights')

    '''
    img = state[0, 0:3]
    img = img.data.cpu().numpy()
    img = img.transpose((1, 2, 0))
    print(img.shape)
    plt.imshow(img)
    plt.savefig(f'{steps}.png')
    '''
    print("Timestep: %d, Action: %d, Reward: %.2f, Q: %.2f, Loss: %.2f" % (steps, action_index, reward, Q_sa[-1], loss))


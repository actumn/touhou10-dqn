import math
import random

import matplotlib.pyplot as plt
import numpy as np
from collections import deque, namedtuple
from skimage import color, transform, exposure
from PIL import Image

from th10.game import TH10
from th10.process import find_process, image_grab
from th10.memory_reader import MemoryReader
from config import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
game = TH10()
memory = deque()


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(IMG_CHANNELS, 32, kernel_size=8, stride=2)
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
        self.head = nn.Linear(128, NUM_OF_ACTIONS)

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


transform = T.Compose([T.Grayscale(), T.Resize((TRANSFORM_HEIGHT, TRANSFORM_WIDTH)), T.ToTensor()])


def transform_state(single_state):
    # PIL -> Grayscale -> Resize -> ToTensor
    single_state = transform(single_state)
    single_state = single_state.unsqueeze(0)
    single_state = single_state.to(device, dtype=torch.float)
    return single_state


policy_net = DQN().to(device)
# policy_net.load_state_dict(torch.load(f'./weights_{NUM_OF_ACTIONS}'))
target_net = DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
"""
test_input = torch.rand(1, 4, 128, 128).to(device, dtype=torch.float)
policy_net(test_input)
"""
optimizer = optim.Adam(policy_net.parameters())
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'is_terminal'))


state, _, _ = game.play(-1)
state = transform_state(state)
state = torch.cat((state, state, state, state), 1)

steps = 0
while True:
    loss = 0
    train_q = torch.tensor([0])

    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps / EPS_DECAY)
    if random.random() <= eps_threshold:
        # choose a random action
        action = torch.tensor([[random.randrange(NUM_OF_ACTIONS)]], device=device, dtype=torch.long)
        q_val = 0
    else:
        # input a stack of 4 images, get the prediction
        q = policy_net(state).max(1)
        action = q[1].view(1, 1)
        q_val = q[0].item()

    next_state, reward, is_terminal = game.play(action.item())
    if next_state is None:
        continue
    next_state = transform_state(next_state)
    next_state = torch.cat((next_state, state[:, :3]), 1)
    reward = torch.tensor([reward], device=device, dtype=torch.float)
    is_terminal = torch.tensor([is_terminal], device=device, dtype=torch.uint8)
    '''
    We need enough states in our experience replay deque so that we can take a random sample from it of the size we declared.
    Therefore we wait until a certain number and observe the environment until we're ready.
    '''
    memory.append((state, action, next_state, reward, is_terminal))
    if len(memory) > EXP_REPLAY_MEMORY:
        memory.popleft()

    # Optimize
    if len(memory) > BATCH_SIZE:
        # Batches
        transitions = random.sample(memory, BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        # Current results
        state_batch = torch.cat(batch.state)
        next_state_batch = torch.cat(batch.next_state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        is_terminal_batch = torch.cat(batch.is_terminal)
        state_action_values = policy_net(state_batch).gather(1, action_batch)

        # Non-final next state
        non_final_mask = is_terminal_batch == 0
        non_final_next_states = next_state_batch[non_final_mask]

        # Non-final next state reward
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

        # Compute the expected Q values
        # (current state reward) + (next state reward) * gamma
        expected_state_action_values = reward_batch + (next_state_values * GAMMA)
        train_q = expected_state_action_values

        # Optimize with mean squared error
        # loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        # Optimize with Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        optimizer.zero_grad()
        loss.backward()
        for param in policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()

    state = next_state
    steps += 1

    if steps % STEPS_SAVE == 0:
        torch.save(policy_net.state_dict(), f'./weights_{NUM_OF_ACTIONS}')

    if steps % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
    '''
    img = state[0, 0:3]
    img = img.data.cpu().numpy()
    img = img.transpose((1, 2, 0))
    plt.imshow(img)
    plt.savefig(f'steps/{steps}.png')
    '''
    print("Timestep: %d, Action: %d, Reward: %.2f, q: %.2f, train_q_min: %.2f, train_q_max: %.2f, Loss: %.2f" %
          (steps, action.item(), reward.item(), q_val, torch.min(train_q), torch.max(train_q), loss))


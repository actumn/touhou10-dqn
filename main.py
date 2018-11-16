from th10.game import TH10
from train import transform_state, DQN
from config import *

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
policy_net = DQN().to(device)
policy_net.load_state_dict(torch.load(f'./weights_{NUM_OF_ACTIONS}'))

game = TH10()
state, _, _ = game.play(-1)
state = transform_state(state)
state = torch.cat((state, state, state, state), 1)

while True:
        q = policy_net(state).max(1)
        action = q[1].view(1, 1)
        next_state, reward, is_terminal = game.play(action.item())

        if next_state is None:
            break

        next_state = transform_state(next_state)
        next_state = torch.cat((next_state, state[:, :3]), 1)
        state = next_state

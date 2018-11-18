from th10.game import TH10
from train import transform_state, DQN
from config import NUM_OF_ACTIONS

import matplotlib.pyplot as plt
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
policy_net = DQN().to(device)
policy_net.load_state_dict(torch.load(f'./weights_{NUM_OF_ACTIONS}'))
policy_net.eval()

game = TH10()
state, _, _ = game.play(-1)
state = transform_state(state)
state = torch.cat((state, state, state, state), 1)

steps = 0
while True:
    q = policy_net(state).max(1)
    action = q[1].view(1, 1)
    next_state, reward, is_terminal = game.play(action.item())

    if next_state is None:
        break

    next_state = transform_state(next_state)
    next_state = torch.cat((next_state, state[:, :3]), 1)
    state = next_state
    print('action: ', action)
    """
    if reward < 0:
        img = state[0, 0:3]
        img = img.data.cpu().numpy()
        img = img / 255.
        img = img.transpose((1, 2, 0))
        plt.imshow(img)
        plt.savefig(f'steps/{steps}.png')
        steps += 1
    """

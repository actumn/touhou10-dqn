import matplotlib.pyplot as plt
from th10.game import TH10

game = TH10()
state, _, _ = game.play(-1)
plt.imshow(state)
plt.show()

import matplotlib.pyplot as plt
from th10.game import TH10

game = TH10()
state, _, _ = game.play(-1)
game.restart_on_end()
'''
game.play(2)
game.play(1)
game.play(2)
game.play(3)
game.play(4)
game.play(0)
'''
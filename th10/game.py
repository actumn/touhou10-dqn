from config import REWARD_IN_ENV, REWARD_DEATH, REWARD_ON_POWER, REWARD_ON_HIT, REWARD_ON_BE_SHOT
from .memory_reader import MemoryReader
from .process import find_process, image_grab, set_foreground
from .directkeys import press_key, release_key, DIK_Z, DIK_LEFT, DIK_RIGHT, DIK_UP, DIK_DOWN, DIK_LSHIFT, DIK_LCONTROL

import numpy as np
import time


def action(action_index):
    if action_index < 0:
        return

    press_key(DIK_Z)
    press_key(DIK_LSHIFT)
    press_key(DIK_LCONTROL)
    release_key(DIK_LEFT)
    release_key(DIK_RIGHT)
    release_key(DIK_UP)
    release_key(DIK_DOWN)
    if action_index == 0:
        return
    elif action_index == 1:
        press_key(DIK_LEFT)
    elif action_index == 2:
        press_key(DIK_RIGHT)
    elif action_index == 3:
        press_key(DIK_UP)
    elif action_index == 4:
        press_key(DIK_DOWN)
    elif action_index == 5:
        press_key(DIK_LEFT)
        press_key(DIK_UP)
    elif action_index == 6:
        press_key(DIK_LEFT)
        press_key(DIK_DOWN)
    elif action_index == 7:
        press_key(DIK_RIGHT)
        press_key(DIK_UP)
    elif action_index == 8:
        press_key(DIK_RIGHT)
        press_key(DIK_DOWN)


class TH10(object):

    def __init__(self):
        pid, hwnd = find_process('th10.exe')
        self.hwnd = hwnd
        self.pid = pid
        self.memory_reader = MemoryReader(pid)
        self.player = self.memory_reader.player_info()

    def play(self, action_index):
        set_foreground(self.hwnd)
        action(action_index)

        prev_powers = self.player.powers
        prev_life = self.player.life
        self.player = self.memory_reader.player_info()
        if self.player.life > 0:
            prev_life = prev_life if prev_life > 0 else 10  # 10 for reset game
            reward = self.calculate_reward(prev_powers, prev_life)
            return image_grab(self.hwnd, (34, 36, -228, -18)), reward, False
        elif self.player.life == prev_life:  # Not on playing game
            self.restart_on_end()
            self.player = self.memory_reader.player_info()
            return None, 0, False
        else:
            self.restart_on_end()
            self.player = self.memory_reader.player_info()
            return image_grab(self.hwnd, (34, 36, -228, -18)), 0, True

    def restart_on_end(self):
        set_foreground(self.hwnd)
        release_key(DIK_Z)
        release_key(DIK_LSHIFT)
        release_key(DIK_LEFT)
        release_key(DIK_RIGHT)
        release_key(DIK_UP)
        release_key(DIK_DOWN)
        press_key(DIK_Z)
        time.sleep(0.2)
        release_key(DIK_Z)
        press_key(DIK_DOWN)
        time.sleep(0.2)
        release_key(DIK_DOWN)
        press_key(DIK_Z)
        time.sleep(0.2)
        release_key(DIK_Z)

    def calculate_reward(self, prev_powers, prev_life):
        enemies = self.memory_reader.enemies_info()
        bullets = self.memory_reader.bullet_info()
        # reward = reward_in_env if self.player.invincible_time <= 0 else 0
        reward = REWARD_IN_ENV
        return reward + \
               REWARD_ON_POWER * (self.player.powers - prev_powers) - \
               REWARD_DEATH * (self.player.life - prev_life) + \
               REWARD_ON_BE_SHOT * self.player.is_near(bullets) + \
               REWARD_ON_HIT * self.player.on_hit(enemies)


from config import reward_in_env, death_reward, reward_on_power, reward_on_dodge, reward_on_hit
from .memory_reader import MemoryReader
from .process import find_process, image_grab, set_foreground
from .directkeys import press_key, release_key, DLK_Z, DLK_LEFT, DLK_RIGHT, DLK_UP, DLK_DOWN

import numpy as np
import time


def action(action_index):
    if action_index < 0:
        return

    press_key(DLK_Z)
    release_key(DLK_LEFT)
    release_key(DLK_RIGHT)
    release_key(DLK_UP)
    release_key(DLK_DOWN)
    if action_index == 0:
        return
    elif action_index == 1:
        press_key(DLK_LEFT)
    elif action_index == 2:
        press_key(DLK_RIGHT)
    elif action_index == 3:
        press_key(DLK_UP)
    elif action_index == 4:
        press_key(DLK_DOWN)
    elif action_index == 5:
        press_key(DLK_LEFT)
        press_key(DLK_UP)
    elif action_index == 6:
        press_key(DLK_LEFT)
        press_key(DLK_DOWN)
    elif action_index == 7:
        press_key(DLK_RIGHT)
        press_key(DLK_UP)
    elif action_index == 8:
        press_key(DLK_RIGHT)
        press_key(DLK_DOWN)


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

        enemies = self.memory_reader.enemies_info()

        prev_powers = self.player.powers
        prev_life = self.player.life if self.player.life > 0 else 10  # 10 for reset game
        self.player = self.memory_reader.player_info()
        if self.player.life > 0:
            reward = self.calculate_reward(prev_powers, prev_life)
            return np.array(image_grab(self.hwnd))[44:488, 34:416], reward, False
        else:
            release_key(DLK_Z)
            release_key(DLK_LEFT)
            release_key(DLK_RIGHT)
            release_key(DLK_UP)
            release_key(DLK_DOWN)
            self.restart_on_end()
            self.player = self.memory_reader.player_info()
            return np.array(image_grab(self.hwnd))[44:488, 34:416], death_reward, True

    def restart_on_end(self):
        set_foreground(self.hwnd)
        press_key(DLK_Z)
        time.sleep(0.2)
        release_key(DLK_Z)
        press_key(DLK_DOWN)
        time.sleep(0.2)
        release_key(DLK_DOWN)
        press_key(DLK_Z)
        time.sleep(0.2)
        release_key(DLK_Z)

    def calculate_reward(self, prev_powers, prev_life):
        enemies = self.memory_reader.enemies_info()
        bullets = self.memory_reader.bullet_info()
        # reward = reward_in_env if self.player.invincible_time <= 0 else 0
        reward = reward_in_env
        return reward + \
               reward_on_power * (self.player.powers - prev_powers) - \
               death_reward * (self.player.life - prev_life) + \
               reward_on_dodge * self.player.is_near(bullets) + \
               reward_on_hit * self.player.on_hit(enemies)


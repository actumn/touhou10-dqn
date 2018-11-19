TRANSFORM_HEIGHT = 128
TRANSFORM_WIDTH = 128
IMG_CHANNELS = 4
BATCH_SIZE = 128  # batch size for update in exp replay
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 2000
GAMMA = 0.999
NUM_OF_ACTIONS = 5  # 9  # stop, left, right, up, down, left-up, left-down, right-up, right-down
REWARD_DEATH = -50  # at the moment the agent stays dead for two frames, so it will receive this reward twice
REWARD_IN_ENV = 1  # reward for living in the environment
REWARD_ON_POWER = 0  # reward for getting power item
REWARD_ON_HIT = 0  # reward for hitting enemy
REWARD_ON_BE_SHOT = -5  # reward for being shot
STEPS_SAVE = 500  # saves weights_9 at these iterations of timesteps
EXP_REPLAY_MEMORY = 7000  # length of exp replay deque before popping values
TARGET_UPDATE = 500

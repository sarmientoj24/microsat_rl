from src.commons.replay_buffer import ReplayBuffer, ReplayBufferLSTM2
from src.commons.normalized_actions import NormalizedActions
from src.commons.seeding import set_seed_everywhere
from src.commons.plot_wandb import WandbLogger
from src.commons.yaml import get_yaml_args
from src.commons.per_buffer import PrioritizedReplayBuffer
import time
import numpy as np


def get_remaining_time(max_iter, start_time, curr_iter):
    current_time = time.time()
    elapsed_time = current_time - start_time
    remaining_time = (elapsed_time / (curr_iter + 1)) * (max_iter - (curr_iter + 1))
    return remaining_time


def get_modified_state(state, to_remove='wbi'):
    if to_remove == 'wbi':
        indices = [0, 1, 2]
    else:
        indices = [3, 4, 5]
    '''
        state: is an np.array of size (50,)
    '''
    new_state = []
    for i in range(len(state)):
        if i % 10 not in indices:
            new_state.append(state[i])
    new_state = np.array(new_state)
    return new_state
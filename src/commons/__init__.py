from src.commons.replay_buffer import ReplayBuffer, ReplayBufferLSTM2
from src.commons.normalized_actions import NormalizedActions
from src.commons.seeding import set_seed_everywhere
from src.commons.plot_wandb import WandbLogger
from src.commons.yaml import get_yaml_args
import time



def get_remaining_time(max_iter, start_time, curr_iter):
    current_time = time.time()
    elapsed_time = current_time - start_time
    remaining_time = (elapsed_time / (curr_iter + 1)) * (max_iter - (curr_iter + 1))
    return remaining_time
import random
import sys
from collections import deque

import gymnasium as gym

from matplotlib import pyplot as plt

import numpy as np
import torch

from env import temp_scale, TEMP_SETPOINT, AllostaticEnv

FILE = sys.argv[1]

SEED = np.random.randint(0, 100000)
print("seed @ ", SEED)
CUDA = False
GPU = 0

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device(f"cuda:{GPU}" if torch.cuda.is_available() and CUDA else "cpu")
if not CUDA:
    torch.set_num_threads(1)
    
    
def make_env():
    item = AllostaticEnv()
    return item

from ppo_small import Agent

env_ = gym.vector.SyncVectorEnv([make_env])

env = make_env()
options = {"from_setpoint": True}

env = gym.wrappers.ClipAction(env)
env = gym.wrappers.RescaleAction(env, 0, 1)  # for Beta policy

env.set_cue_probs(p_on=0.005)

print(env.observation_space)

agent = Agent(env_).to(device)
if "saved_models" in FILE:
    agent.load_state_dict(torch.load(FILE, map_location=torch.device('cpu')))
else:
    agent.load_state_dict(torch.load("saved_models/" + FILE, map_location=torch.device('cpu')))

TEST_CUE = True
N = 500

hist_state = deque()
hist_cue = deque()
hist_action = deque()
hist_load = deque()
hist_memory = deque()

global_step = 0
next_obs, info = env.reset(options=options)
next_obs = torch.Tensor(next_obs[None]).to(device)

done = False
while not done:

    # ALGO LOGIC: action logic
    with torch.no_grad():
        action, _, _, value = agent.get_action_and_value(next_obs)
    
    # TRY NOT TO MODIFY: execute the game and log data.
    action_ = action.cpu().numpy()[0]
    
    next_obs, reward, done, truncated, info = env.step(action_)
    
    hist_state.append(info["temp_c"])
    hist_cue.append(info["cue"])
    hist_action.append(info["q_control"])
    hist_load.append(info["load"])
    hist_memory.append(info["memory"])

    done = done | truncated
    next_obs = torch.Tensor(next_obs[None]).to(device)
    global_step += 1
    
    if global_step > N:
        done = True
    
env.close()

plt.figure()
hists = [hist_state, hist_cue, hist_action, hist_load, hist_memory]
legends = ["temp", "cue", "control", "load", "memory"]

# save data as json
import json
data = {k: list(v) for k, v in zip(legends, hists)}
with open("sample_data/allostasis.json", mode="w") as f:
    json.dump(data, f)

for i, hist in enumerate(hists):
    plt.subplot(len(legends), 1, i + 1)
    plt.plot(hist, alpha=0.5)
    plt.legend([legends[i]])

plt.tight_layout()
plt.show()

print(f"{FILE} finish. @ {global_step} steps")

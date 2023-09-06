import random
import sys
from collections import deque

import gymnasium as gym

from matplotlib import pyplot as plt

import numpy as np
import torch

from env import AllostaticEnv, TEMP_CUE_OFF, temp_scale, TEMP_SETPOINT

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

env = AllostaticEnv()

init_temp = np.array([40] * 8) + 3 * np.random.rand(8)
init_temp[2] = 50
options = {"initial_temp": init_temp}

env = gym.wrappers.ClipAction(env)
env = gym.wrappers.RescaleAction(env, 0, 1)  # for Beta policy

env.set_cue_probs(p_on=0.001, p_off=0.01)

print(env.observation_space)

agent = Agent(env_).to(device)
if "saved_models" in FILE:
    agent.load_state_dict(torch.load(FILE, map_location=torch.device('cpu')))
else:
    agent.load_state_dict(torch.load("saved_models/" + FILE, map_location=torch.device('cpu')))

agent.eval()

N = 2000
cue_hist = deque(maxlen=N)
temp_hist = deque(maxlen=N)
action_hist = deque(maxlen=N)
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
    
    cue_hist.append(next_obs[1])
    temp_hist.append(next_obs[0])
    action_hist.append(action_[0])

    done = done | truncated
    next_obs = torch.Tensor(next_obs[None]).to(device)
    global_step += 1
    
    if global_step > N:
        done = True
    
env.close()

action_hist = np.array(action_hist)
cue_hist = np.array(cue_hist)
temp_hist = np.array(temp_hist)

pseudo_setpoint = env.get_pseudo_setpoint(action_hist)

np.save("sample_data/action_hist", action_hist)
np.save("sample_data/cue_hist", cue_hist)
np.save("sample_data/temp_hist", temp_hist)
np.save("sample_data/pseudo_setpoint", pseudo_setpoint)

plt.figure()
plt.clf()
plt.plot(cue_hist)
plt.plot(temp_hist)
plt.plot(np.ones_like(temp_hist) * temp_scale(TEMP_SETPOINT + 273), "--")
plt.plot(np.ones_like(temp_hist) * temp_scale(TEMP_CUE_OFF + 273), "--")
plt.plot(action_hist, alpha=0.3)
# plt.plot(pseudo_setpoint, alpha=0.3)
plt.ylabel("Normalized Obs")
plt.xlabel("Step")
plt.legend(["cue", "temp", "setpoint", "cue_off", "action"])#, "pseudo_setpoint"])
plt.show()

print(f"{FILE} finish. @ {global_step} steps")

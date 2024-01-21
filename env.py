from collections import deque
from copy import deepcopy

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box

MAX_TIMESTEP = 10_000
TEMP_MAX = 42.
TEMP_MIN = 32.
TEMP_SETPOINT = 36.
TEMP_CUE_OFF = 38.
TEMP_BG = 25.

def absolute2celsius(temp_absolute):
    return temp_absolute - 273.

def temp_scale(temp):
    return (absolute2celsius(temp) - TEMP_SETPOINT) / (TEMP_MAX - TEMP_MIN)


class AllostaticEnv(gym.Env):
    def __init__(self):
        super().__init__()
        
        self.action_space = Box(low=0, high=1, shape=(1,))
        self.observation_space = Box(low=0, high=1, shape=(2,))
        
        self.episode_length = MAX_TIMESTEP
        self.steps = 0

        self.temp_target = 273. + TEMP_SETPOINT
        self.temp = 273. + TEMP_SETPOINT
        self.temp_prev = 273 + TEMP_SETPOINT

        self.cue = False
        self.cue_prev = False
        self.prob_cue_on = 0.001

        # decaying cue memory
        self.cue_memory = 0.0
        self.decay_cue = 0.95
        
        # temperature reservoir model
        self.dt = 1
        self.temp_background = 273 + TEMP_BG
        self.k_heat_gen = 5
        self.capacitance = 30
        self.resistance = 10
        
        self.count_after_cue_presentation = None
        self.thermal_load = 10.0  # external thermal load following the cue signal
        self.step_delay_thermal_load = 20  # steps after the cue presentation
        self.step_length_thermal_load = 50 # length of thermal load steps
        
        self.drive_coef_temp = 100.
        self.action_cost = 0.05
        
    def set_cue_probs(self, p_on):
        self.prob_cue_on = p_on
        
    def get_pseudo_setpoint(self, action):
        return self.temp_background + self.resistance * self.k_heat_gen * action
 
    def reset(self, *, seed, options):
        super().reset(seed=seed, options=options)
        self.steps = 0
        
        from_setpoint = None
        if options is not None:
            from_setpoint = options.get("from_setpoint")
        
        if from_setpoint is True:
            self.temp = 273. + TEMP_SETPOINT
        else:
            self.temp = 273. + TEMP_SETPOINT + np.random.randn()

        self.temp_prev = 273 + TEMP_SETPOINT
        self.cue = False
        self.cue_prev = False

        self.count_after_cue_presentation = None
        self.cue_memory = 0.0

        info = {
            "temp_c": absolute2celsius(self.temp),
            "cue": self.cue,
            "memory": self.cue_memory,
            "load": 0,
            "interoception": np.array([temp_scale(self.temp)])
        }
        return self.get_current_obs(), info
        
    def step(self, action):
        
        thermal_action = 2 * action - 1
        
        self.temp_prev = deepcopy(self.temp)
        self.cue_prev = deepcopy(self.cue)
        
        q_load, q_control = self.thermal_model_update(thermal_action)
        self.cue_memory_update()
        self.steps += 1

        obs = self.get_current_obs()
        reward = self.get_reward(thermal_action[0])
        
        done = self.steps > self.episode_length
        done = done or absolute2celsius(self.temp) < TEMP_MIN or TEMP_MAX < absolute2celsius(self.temp)

        info = {
            "temp_c": absolute2celsius(self.temp),
            "cue": self.cue,
            "memory": self.cue_memory,
            "load": q_load,
            "interoception": np.array([temp_scale(self.temp)]),
            "q_control": q_control,
        }

        return obs, reward, done, False, info
    
    def thermal_model_update(self, action):
        load_start = self.step_delay_thermal_load
        load_end = self.step_length_thermal_load + self.step_delay_thermal_load

        q_load = 0.0
        if self.count_after_cue_presentation is not None:
            if load_start <= self.count_after_cue_presentation < load_end:
                q_load = self.thermal_load
        
            if load_end <= self.count_after_cue_presentation:
                self.count_after_cue_presentation = None
        
        q_out = (self.temp - self.temp_background) / self.resistance

        # controllable and uncontrollable heat factors
        q_controllable = self.k_heat_gen * action[0]
        q_uncontrollable = q_load - q_out

        q_total = q_controllable + q_uncontrollable
        self.temp += self.dt * q_total / self.capacitance
        
        return q_load, q_controllable
        
    def cue_memory_update(self):
        if self.count_after_cue_presentation is not None:
            self.count_after_cue_presentation += 1

        self.cue = False
        if np.random.rand() < self.prob_cue_on:
            self.cue = True
            self.count_after_cue_presentation = 0

        self.cue_memory = self.decay_cue * self.cue_memory + float(self.cue)

    def get_reward(self, action):

        def drive(temp, cue):
            return self.drive_coef_temp * temp_scale(temp) ** 2

        r = drive(self.temp_prev, self.cue_prev) - drive(self.temp, self.cue)
        # r += - self.action_cost * action ** 2
        return r
    
    def get_current_obs(self):
        return np.array([temp_scale(self.temp), self.cue_memory], dtype=np.float32)


class SickEnv(gym.Env):
    def __init__(self):
        super().__init__()
        
        self.action_space = Box(low=0, high=1, shape=(1,))
        self.observation_space = Box(low=0, high=1, shape=(2,))
        
        self.episode_length = MAX_TIMESTEP
        self.steps = 0

        self.temp_target = 273. + TEMP_SETPOINT
        self.temp = 273. + TEMP_SETPOINT
        self.temp_prev = 273 + TEMP_SETPOINT
        
        self.cue = False
        self.cue_prev = False
        # self.prob_cue_on = 0.01
        # self.prob_cue_off = 0.1
        self.prob_cue_on = 0.001
        self.prob_cue_off = 0.1

        self.prob_death_at_sick = 0.0001
        
        # decaying cue memory
        self.cue_memory = 0.0
        self.decay_cue = 0.95

        # temperature reservoir model
        self.dt = 1
        self.temp_background = 273 + TEMP_BG
        self.k_heat_gen = 10
        self.capacitance = 30
        self.resistance = 10
        
        self.drive_coef_temp = 100.
        self.drive_coef_cue = 1.
        self.action_cost = 0.05
    
    def set_cue_probs(self, p_on, p_off):
        self.prob_cue_on = p_on
        self.prob_cue_off = p_off
    
    def get_pseudo_setpoint(self, action):
        return self.temp_background + self.resistance * self.k_heat_gen * action
    
    def reset(self, *, seed, options):
        super().reset(seed=seed, options=options)
        self.steps = 0
        
        from_setpoint = None
        if options is not None:
            from_setpoint = options.get("from_setpoint")
        
        if from_setpoint is True:
            self.temp = 273. + TEMP_SETPOINT
        else:
            self.temp = 273. + TEMP_SETPOINT + np.random.randn()
        
        self.temp_prev = 273 + TEMP_SETPOINT
        self.cue = False
        self.cue_prev = False
        
        self.count_after_cue_presentation = None
        self.cue_memory = 0.0

        info = {
            "temp_c": absolute2celsius(self.temp),
            "cue": self.cue,
            "memory": self.cue_memory,
            "load": 0,
            "interoception": np.array([temp_scale(self.temp)])
        }
        return self.get_current_obs(), info
    
    def step(self, action):
        
        thermal_action = 2 * action - 1

        self.temp_prev = deepcopy(self.temp)
        self.cue_prev = deepcopy(self.cue)
        
        q_control = self.thermal_model_update(thermal_action)
        self.cue_memory_update()
        self.steps += 1
        
        obs = self.get_current_obs()
        reward = self.get_reward(action[0])
        
        done = self.steps > self.episode_length
        done = done or absolute2celsius(self.temp) < TEMP_MIN or TEMP_MAX < absolute2celsius(self.temp)
        
        # if self.cue is True:
        #     done = done or np.random.rand() < self.prob_death_at_sick
        
        info = {
            "temp_c": absolute2celsius(self.temp),
            "cue": self.cue,
            "memory": self.cue_memory,
            "interoception": np.array([temp_scale(self.temp)]),
            "q_control": q_control,
        }
        
        return obs, reward, done, False, info
    
    def thermal_model_update(self, action):
        q_controllable = self.k_heat_gen * action[0]
        # q_controllable = np.clip(q_controllable, a_min=-self.k_heat_gen/2, a_max=self.k_heat_gen)
        
        q_out = (self.temp - self.temp_background) / self.resistance
        q_total = q_controllable - q_out
        self.temp += self.dt * q_total / self.capacitance
        
        return q_controllable
        
    def cue_memory_update(self):
        if self.cue is True:
            if absolute2celsius(self.temp) > TEMP_CUE_OFF:
                if np.random.rand() < self.prob_cue_off:
                    self.cue = False
        else:
            s = np.random.rand()
            if s < self.prob_cue_on:
                self.cue = True
        
        self.cue_memory = self.decay_cue * self.cue_memory + float(self.cue)
    
    def get_reward(self, action):
        
        def drive(temp, cue):
            return self.drive_coef_temp * temp_scale(temp) ** 2
        
        r = drive(self.temp_prev, self.cue_prev) - drive(self.temp, self.cue)
        r += - self.drive_coef_cue * self.cue  # cue penalty
        r += - self.action_cost * action
        return r
    
    def get_current_obs(self):
        return np.array([temp_scale(self.temp), self.cue], dtype=np.float32)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # env = AllostaticEnv()
    env = SickEnv()
    
    done = False
    obs, info = env.reset(seed=0, options={})
    print(obs, info)
    
    hist_state = deque()
    hist_cue = deque()
    hist_action = deque()
    hist_load = deque()
    hist_memory = deque()
    
    while not done:
        action = env.action_space.sample() + 0.01
        obs, reward, done, _, info = env.step(action=action)
        
        if info["cue"] is True:
            print("cue")
        
        hist_state.append(info["temp_c"])
        hist_cue.append(10 * info["cue"])
        hist_action.append(info["q_control"])
        if "load" in info.keys():
            hist_load.append(info["load"])
        hist_memory.append(10 * info["memory"])
        
        # print(obs, reward, action, info)
        
    plt.figure()
    if "load" in info.keys():
        hists = [hist_state, hist_cue, hist_action, hist_load, hist_memory]
        legends = ["temp", "cue", "control", "load", "memory"]
    else:
        hists = [hist_state, hist_cue, hist_action, hist_memory]
        legends = ["temp", "cue", "control", "memory"]

    for i, hist in enumerate(hists):
        plt.subplot(len(legends), 1, i+1)
        plt.plot(hist, alpha=0.5)
        plt.legend([legends[i]])

    plt.show()

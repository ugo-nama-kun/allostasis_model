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
        
        self.temp_target = 273. + TEMP_SETPOINT
        self.temp = 273. + TEMP_SETPOINT
        
        self.cue = False
        # self.prob_cue_on = 0.01
        # self.prob_cue_off = 0.1
        self.prob_cue_on = 0.001
        self.prob_cue_off = 0.01

        self.temp_prev = 273 + TEMP_SETPOINT
        self.cue_prev = False

        self.episode_length = MAX_TIMESTEP
        self.steps = 0
        
        # thermal params
        self.dt = 1
        self.temp_background = 273 + TEMP_BG
        self.k_heat_gen = 10
        self.capacitance = 30
        self.resistance = 10
        
        self.drive_coef_temp = 100.
        self.drive_coef_cue = 0.3
        self.action_cost = 0.05
        
    def set_cue_probs(self, p_on, p_off):
        self.prob_cue_on = p_on
        self.prob_cue_off = p_off
        
    def get_pseudo_setpoint(self, action):
        return temp_scale(self.temp_background + self.resistance * self.k_heat_gen * action)
        
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
        info = {"temp_c": absolute2celsius(self.temp), "cue": self.cue}
        return self.get_current_obs(), info
        
    def step(self, action):
        
        self.temp_prev = deepcopy(self.temp)
        self.cue_prev = deepcopy(self.cue)
        
        self.thermal_model_update(action)
        self.cue_update()
        self.steps += 1

        obs = self.get_current_obs()
        reward = self.get_reward(action[0])
        
        done = self.steps > self.episode_length
        done = done or absolute2celsius(self.temp) < TEMP_MIN or TEMP_MAX < absolute2celsius(self.temp)

        # death by cue
        # death = False
        # if self.cue is True and np.random.rand() < self.prob_cue_on:
        #     death = True
        # done = done or death

        info = {"temp_c": absolute2celsius(self.temp), "cue": self.cue, "interoception": np.array([temp_scale(self.temp)])}

        return obs, reward, done, False, info
    
    def thermal_model_update(self, action):
        q_in = self.k_heat_gen * action[0]
        q_out = (self.temp - self.temp_background) / self.resistance
        q_diff = q_in - q_out
        self.temp += self.dt * q_diff / self.capacitance
        
    def cue_update(self):
        if self.cue is True:
            if absolute2celsius(self.temp) > TEMP_CUE_OFF:
                if np.random.rand() < self.prob_cue_off:
                    self.cue = False
        else:
            s = np.random.rand()
            if s < self.prob_cue_on:
                self.cue = True

    def get_reward(self, action):

        def drive(temp, cue):
            return self.drive_coef_temp * temp_scale(temp) ** 2

        r = drive(self.temp_prev, self.cue_prev) - drive(self.temp, self.cue)
        # r = - drive(self.temp, self.cue)
        r += - self.drive_coef_cue * self.cue  # cue penalty
        r += - self.action_cost * action
        return r
    
    def get_current_obs(self):
        return np.array([temp_scale(self.temp), self.cue], dtype=np.float32)


if __name__ == '__main__':
    env = AllostaticEnv()
    
    done = False
    obs, info = env.reset(seed=0, options={})
    print(obs, info)
    
    while not done:
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action=action)
        
        print(obs, reward, action, info)

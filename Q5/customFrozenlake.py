from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv
from gymnasium.envs.toy_text.utils import categorical_sample

import numpy as np


class CustomFrozenLakeEnv(FrozenLakeEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def step(self, a):
        transitions = self.P[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, t = transitions[i]
        # Adds a 2% chance of death of a cold
        if np.random.rand() < 0.02:
            s = self.s  # Agent stays in the same state
            r = 0.0  # No reward for dying
            t = True  # end the episode
        self.s = s
        self.lastaction = a
        if self.render_mode == "human":
            self.render()
        return (int(s), r, t, False, {"prob": p})

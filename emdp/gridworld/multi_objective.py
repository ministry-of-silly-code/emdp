import random
import typing

import numpy as np

import emdp
from . import GridWorldMDP

STATE = typing.Tuple[np.array, int]


class MultiObjectiveGridWorldMDP(emdp.MDP):
    def __init__(self, gws: typing.List[GridWorldMDP]):
        g0 = gws[0]
        rewards = np.stack([g.reward for g in gws], -1)
        self.gws = gws
        self.current_gw = gws[0]
        self.num_tasks = len(gws)
        super().__init__(g0.transition, rewards, g0.discount, g0.initial_state, terminal_states=g0.terminal_states)

    @property
    def observation_space(self):
        return self.gws[0].observation_space

    @property
    def action_space(self):
        return self.gws[0].action_space

    def seed(self, *args, **kwargs):
        for g in self.gws:
            g.seed(*args, **kwargs)

    def reset(self) -> STATE:
        self.current_gw = random.randint(0, self.num_tasks - 1)
        return super().reset(), self.current_gw

    def plot_sa(self, *args, **kwargs):
        return self.gws[self.current_gw].plot_sa(*args, **kwargs)

    def step(self, *args, **kwargs) -> (STATE, int, bool, dict):
        state, reward, done, info = super().step(*args, **kwargs)
        return (state, self.current_gw), reward[self.current_gw], done, info


def four_rooms_gw_multiobjective(goals, initial_states):
    gws = [GridWorldMDP(goal, initial_states=initial_states) for goal in goals]  # TODO: initial state
    gw = MultiObjectiveGridWorldMDP(gws)
    return gw

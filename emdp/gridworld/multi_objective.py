import typing

import numpy as np

from . import MDP, GridWorldMDP


class MultiObjectiveGridWorldMDP(MDP):
    def __init__(self, gws: typing.List[MDP]):
        g0 = gws[0]
        rewards = np.stack([g.reward for g in gws], -1)
        self.gws = gws
        super().__init__(g0.transition, rewards, g0.discount, g0.initial_state, terminal_states=g0.terminal_states)


def four_rooms_gw_multiobjective(goals):
    gws = [GridWorldMDP(goal) for goal in goals]
    gw = MultiObjectiveGridWorldMDP(gws)
    return gw

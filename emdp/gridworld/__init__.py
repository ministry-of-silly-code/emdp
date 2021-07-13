"""
A simple grid world environment
"""
import gym.spaces
import matplotlib.pyplot as plt
import numpy as np

import emdp.actions
from . import builder_tools
from . import txt_utilities
from .helper_utilities import flatten_state, unflatten_state
from ..common import MDP


class GridWorldMDP(MDP):
    def __init__(self, goal=None, initial_states=None, ascii_room=None, goals=None, seed=1337):
        assert (goal and not goals) or (not goal and goals)

        if ascii_room is None:
            ascii_room = """
            #########
            #   #   #
            #       #
            #   #   #
            ## ### ##
            #   #   #
            #       #
            #   #   #
            #########"""[1:].split('\n')
        ascii_room = [row.strip() for row in ascii_room]

        terminal_states = (goal,)
        char_matrix = txt_utilities.get_char_matrix(ascii_room)
        grid_size = len(char_matrix[0])
        builder = builder_tools.TransitionMatrixBuilder(
            grid_size=grid_size,
            action_space=4,
            terminal_states=terminal_states,
            p_success=1.
        )

        walls, empty_coords = txt_utilities.ascii_to_walls(char_matrix)  # hacks
        self.reachable_states_idx = []
        for e in empty_coords:
            (e,), = flatten_state(e, grid_size, grid_size * grid_size).nonzero()
            self.reachable_states_idx.append(e)

        for (r, c) in walls:
            builder.add_wall_at((r, c))

        reward = np.zeros(builder.P.shape[:2], dtype=np.float32)

        idx = lambda r, c: flatten_state((r, c), builder.grid_size, builder.P.shape[0]).argmax()

        # left = np.array(goal) + emdp.actions.LEFT_vec
        # right = np.array(goal) + emdp.actions.RIGHT_vec
        # up = np.array(goal) + emdp.actions.UP_vec
        # down = np.array(goal) + emdp.actions.DOWN_vec
        # reward[idx(*left), emdp.actions.RIGHT] = 1
        # reward[idx(*right), emdp.actions.LEFT] = 1
        # reward[idx(*up), emdp.actions.DOWN] = 1
        # reward[idx(*down), emdp.actions.UP] = 1

        reward[idx(*goal), :] = 1

        initial_state_distr = np.zeros(reward.shape[0])

        if initial_states is None:
            initial_state_distr[self.reachable_states_idx] = 1 / len(self.reachable_states_idx)
        else:
            for e in initial_states:
                (e,), = flatten_state(e, grid_size, grid_size * grid_size).nonzero()
                initial_state_distr[e] = 1 / len(initial_states)

        discount, size = 0.9, builder.grid_size

        terminal_states = list(map(lambda tupl: int(size * tupl[0] + tupl[1]), terminal_states))
        self.size = size
        super().__init__(builder.P, reward, discount, initial_state_distr, terminal_states, seed=seed)
        self.action_space = gym.spaces.Discrete(self.num_actions)
        self.observation_space = gym.spaces.Box(low=0., high=1., shape=(self.num_states,))

    # def reset(self):
    #     super().reset()
    #     return self.current_state

    def flatten_state(self, state):
        """Flatten state (x,y) into a one hot vector"""
        return flatten_state(state, self.size, self.num_states)

    def unflatten_state(self, onehot):
        """Unflatten a one hot vector into a (x,y) pair"""
        return unflatten_state(onehot, self.size)

    # def step(self, action):
    #     state, reward, done, info = super().step(action)
    #     return state, reward, done, info

    def set_current_state_to(self, tuple_state):
        return super().set_current_state_to(self.flatten_state(tuple_state).argmax())

    def plot_sa(self, title, data, scale_data=True, frame=(0, 0, 0, 0)):
        """ This is going to generate a quiver plot to visualize the policy graphically.
        It is useful to see all the probabilities assigned to the four possible actions in each state """

        assert data.shape == (self.num_states, self.num_actions)
        data_reachable = np.zeros_like(data)
        scale = np.abs(data).max()

        data_unreachable = data.copy()

        data_reachable[self.reachable_states_idx, :] = data[self.reachable_states_idx, :]
        data_unreachable[self.reachable_states_idx, :] = 0.
        del data

        if scale_data:
            data_reachable = data_reachable / (scale * 1.1)
            data_unreachable = np.clip(data_unreachable / (scale * 1.1), -1, 1)

        num_cols, num_rows = self.size, self.size
        num_states, num_actions = data_unreachable.shape
        assert num_actions == 4

        direction = np.zeros((num_actions, 2))
        direction[emdp.actions.UP] = emdp.actions.UP_vec
        direction[emdp.actions.DOWN] = emdp.actions.DOWN_vec
        direction[emdp.actions.LEFT] = emdp.actions.LEFT_vec
        direction[emdp.actions.RIGHT] = emdp.actions.RIGHT_vec

        c, r = np.meshgrid(np.arange(num_rows), np.arange(num_cols))
        r, c = r.flatten(), c.flatten()
        figure = plt.figure()
        ax = plt.gca()

        for (r_, c_), a in zip(direction, range(num_actions)):
            base = (r_, -c_)
            quivers = np.einsum("d,m->md", base, data_unreachable[:, a])

            pos = data_unreachable[:, a] > 0
            ax.quiver(c[pos], r[pos], *quivers[pos].T, units='xy', scale=2.0, color='g', alpha=0.1)

            pos = data_unreachable[:, a] < 0
            ax.quiver(c[pos], r[pos], *-quivers[pos].T, units='xy', scale=2.0, color='r', alpha=0.1)

        for (r_, c_), a in zip(direction, range(num_actions)):
            base = (r_, -c_)  # TODO: WHY is the flip necessary?
            quivers = np.einsum("d,m->md", base, data_reachable[:, a])

            pos = data_reachable[:, a] > 0
            ax.quiver(c[pos], r[pos], *quivers[pos].T, units='xy', scale=2.0, color='g', alpha=1.0)

            pos = data_reachable[:, a] < 0
            ax.quiver(c[pos], r[pos], *-quivers[pos].T, units='xy', scale=2.0, color='r', alpha=1.0)

        x0, x1, y0, y1 = frame
        # set axis limits / ticks / etc... so we have a nice grid overlay
        ax.set_xlim((x0 - 0.5, num_cols - x1 - 0.5))
        ax.set_ylim((y0 - 0.5, num_rows - y1 - 0.5)[::-1])

        # ax.set_xticks(xs)
        # ax.set_yticks(ys)

        # major ticks

        ax.set_xticks(np.arange(x0, num_cols - x1, 1))
        ax.xaxis.set_tick_params(labelsize=5)
        ax.set_yticks(np.arange(y0, num_rows - y1, 1))
        ax.yaxis.set_tick_params(labelsize=5)

        # minor ticks
        ax.set_xticks(np.arange(*ax.get_xlim(), 1), minor=True)
        ax.set_yticks(np.arange(*ax.get_ylim()[::-1], 1), minor=True)

        ax.grid(which='minor', color='gray', linestyle='-', linewidth=1)
        ax.set_aspect(1)

        tag = f"{title}"

        if scale_data:
            title += f"_{scale:.4f}"

        ax.set_title(title, fontdict={'fontsize': 8, 'fontweight': 'medium'})
        return tag, figure

    def seed(self, seed):
        self.set_seed(seed)

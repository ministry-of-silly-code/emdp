import random

import gym.spaces
import matplotlib.pyplot as plt
import numpy as np
import tqdm

import emdp.actions
import emdp.gridworld.plotting
import emdp.gridworld.plotting
from . import builder_tools
from . import txt_utilities
from .helper_utilities import flatten_state, unflatten_state
from ..common import MDP

TILE_PIXELS = 32


class GridWorldMDP(MDP, gym.Env):
    rewarding_action = [emdp.actions.RIGHT, emdp.actions.DOWN, emdp.actions.LEFT, emdp.actions.UP]
    discount = 0.9
    metadata = {"render.modes": ["ansi", "rgb_array"]}
    ansi_to_rgb = {
        ' ': (255, 255, 255),
        '#': (0, 0, 0),
        'G': (0, 0, 255),
        '@': (0, 255, 0),
    }

    def __init__(self, goal=None, initial_states=None, ascii_room=None, goals=None, seed=1337, strip=True, dict_observations=False, forced_goal=None):
        assert (goal and not goals) or (not goal and goals)
        if goal:
            goals = [goal, ]

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
        if strip:
            ascii_room = [row.strip() for row in ascii_room]
        self.ascii_room = ascii_room
        self.numpy_room = np.array([list(l) for l in self.ascii_room])
        char_matrix = txt_utilities.get_char_matrix(ascii_room)
        self.size = len(char_matrix[0])
        walls, empty_coords = txt_utilities.ascii_to_walls(char_matrix)  # hacks
        self.reachable_states_idx = []
        for e in empty_coords:
            (e,), = flatten_state(e, self.size, self.size * self.size).nonzero()
            self.reachable_states_idx.append(e)

        self.forced_goal = forced_goal
        self.goals = goals

        self.num_tasks = len(goals)
        self.initial_states = initial_states
        self.walls = walls
        self.num_states = self.size * self.size

        self.initial_state_distr = np.zeros(self.num_states)
        if self.initial_states is None:
            self.initial_state_distr[self.reachable_states_idx] = 1 / len(self.reachable_states_idx)
        else:
            for e in self.initial_states:
                (e,), = flatten_state(e, self.size, self.size * self.size).nonzero()
                self.initial_state_distr[e] = 1 / len(self.initial_states)

        self.rewards, self.terminal_matrices, self.transitions = [], [], []
        for goal in goals:
            reward, terminal_matrix, transition = self.rebuild_mdp(goal)
            self.rewards.append(reward)
            self.terminal_matrices.append(terminal_matrix)
            self.transitions.append(transition)

        self.goal = self.pick_goal()
        goal_idx = self.goals.index(self.goal)
        self.reward = self.rewards[goal_idx]
        self.transition = self.transitions[goal_idx]
        self.terminal_matrix = self.terminal_matrices[goal_idx]

        self.initial_seed = seed
        self.dict_observations = dict_observations
        super().__init__(transition, reward, self.discount, self.initial_state_distr, terminal_matrix, seed=seed)
        self.action_space = gym.spaces.Discrete(self.num_actions)
        if dict_observations:
            self.observation_space = gym.spaces.Dict({
                'image': gym.spaces.Box(low=0, high=255, shape=(3, self.size, self.size), dtype="uint8"),
                'task': gym.spaces.Box(low=0., high=1., shape=(self.num_tasks,), dtype=int),
            })
        else:
            self.observation_space = gym.spaces.Box(low=0., high=1., shape=(self.num_states,), dtype=int)
        self.reset()

    def pick_goal(self):
        if self.forced_goal is not None:
            return self.goals[self.forced_goal]
        else:
            return random.choice(self.goals)

    def rebuild_mdp(self, goal):
        builder = builder_tools.TransitionMatrixBuilder(
            grid_size=self.size,
            action_space=4,
            p_success=1.
        )
        for (r, c) in self.walls:
            builder.add_wall_at((r, c))

        reward = np.zeros(builder.P.shape[:2], dtype=np.float32)
        reward[self.flatten_state(goal).argmax(), self.rewarding_action] = 1
        terminal_matrix = reward == 1
        return reward, terminal_matrix, builder.P

    def reset(self):
        self.goal = self.pick_goal()
        goal_idx = self.goals.index(self.goal)

        self.reward = self.rewards[goal_idx]
        self.transition = self.transitions[goal_idx]
        self.terminal_matrix = self.terminal_matrices[goal_idx]
        return super().reset()

    def flatten_state(self, state):
        """Flatten state (x,y) into a one hot vector"""
        return flatten_state(state, self.size, self.num_states)

    def unflatten_state(self, onehot):
        """Unflatten a one hot vector into a (x,y) pair"""
        return unflatten_state(onehot, self.size)

    def set_current_state_to(self, tuple_state):
        return super().set_current_state_to(self.flatten_state(tuple_state).argmax())

    def plot_sa(self, title, data, scale_data=True):
        """ This is going to generate a quiver plot to visualize the policy graphically.
        It is useful to see all the probabilities assigned to the four possible actions in each state """

        assert data.shape[0] == self.num_states  # self.num_actions
        assert len(data.shape) == 2

        data_reachable = np.zeros_like(data)
        data_unreachable = np.copy(data)

        data_reachable[self.reachable_states_idx, :] = data[self.reachable_states_idx, :]
        data_unreachable[self.reachable_states_idx, :] = 0.
        del data
        scale = np.abs(data_reachable).max()

        if scale_data and scale != 0:
            data_reachable = data_reachable / (scale * 1.1)
            data_unreachable = np.clip(data_unreachable / (scale * 1.1), -1, 1)

        num_cols, num_rows = self.size, self.size
        num_states, num_actions = data_unreachable.shape
        np.meshgrid(np.arange(num_rows), np.arange(num_cols))
        c, r = np.meshgrid(np.arange(num_rows), np.arange(num_cols))
        r, c = r.flatten(), c.flatten()
        figure = plt.figure()
        ax = plt.gca()
        self._set_gridworld_frame(ax)

        if num_actions == 4:
            direction = np.zeros((num_actions, 2))
            direction[emdp.actions.LEFT] = emdp.actions.LEFT_xy
            direction[emdp.actions.RIGHT] = emdp.actions.RIGHT_xy
            direction[emdp.actions.UP] = emdp.actions.UP_xy
            direction[emdp.actions.DOWN] = emdp.actions.DOWN_xy

            for (x, y), a in zip(direction, emdp.actions.ACTIONS_ORDER):
                base_xy = (x, -y)
                quivers = np.einsum("d,m->md", base_xy, data_unreachable[:, a])

                pos = data_unreachable[:, a] > 0
                ax.quiver(c[pos], r[pos], *quivers[pos].T, units='xy', scale=2.0, color='g', alpha=0.1)

                pos = data_unreachable[:, a] < 0
                ax.quiver(c[pos], r[pos], *-quivers[pos].T, units='xy', scale=2.0, color='r', alpha=0.1)

            for (x, y), a in zip(direction, range(num_actions)):
                base_xy = (x, -y)  # :(
                quivers = np.einsum("d,m->md", base_xy, data_reachable[:, a])

                pos = data_reachable[:, a] > 0
                U, V = quivers[pos][:, 0], quivers[pos][:, 1]
                ax.quiver(c[pos], r[pos], U, V, units='xy', scale=2.0, color='g', alpha=1.0)

                pos = data_reachable[:, a] < 0
                U, V = -quivers[pos][:, 0], -quivers[pos][:, 1]
                ax.quiver(c[pos], r[pos], U, V, units='xy', scale=2.0, color='r', alpha=1.0)

        else:
            best_actions = data_reachable.argmax(axis=1)
            sizes = 24 * data_reachable[range(data_reachable.shape[0]), best_actions]
            for idx, (a, size) in enumerate(zip(best_actions, sizes)):
                if size > 1:
                    r_, c_ = r[idx], c[idx]
                    # if a < 4:
                    #     a = emdp.actions.ACTIONS_HUMAN[a]
                    ax.text(c_, r_, a, size=size)

        tag = f"{title}"

        if scale_data:
            title += f"_{scale:.4f}"

        ax.set_title(title, fontdict={'fontsize': 8, 'fontweight': 'medium'})
        figure.tight_layout()
        return tag, figure

    def _set_gridworld_frame(self, ax):
        num_cols, num_rows = self.size, self.size
        ax.set_xlim((-0.5, num_cols - 0.5))
        ax.set_ylim((-0.5, num_rows - 0.5)[::-1])

        # major ticks
        ax.set_xticks(np.arange(0, num_cols, 1))
        ax.xaxis.set_tick_params(labelsize=10)
        ax.set_yticks(np.arange(0., num_rows, 1))
        ax.yaxis.set_tick_params(labelsize=10)

        # # minor ticks
        ax.set_xticks(np.arange(*ax.get_xlim(), 1), minor=True)
        ax.set_yticks(np.arange(*ax.get_ylim()[::-1], 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=1)
        ax.set_aspect(1)

    def plot_s(self, title, vf, vmin=0, vmax=1):
        x0, x1, y0, y1 = 0, 0, 0, 0
        vf = vf.reshape(self.size, self.size)
        num_cols, num_rows = vf.shape
        vmin = min(vmin, vf.min())
        vmax = max(vmax, vf.max())

        figure = plt.figure()
        ax = plt.gca()

        ax.set_xlim((x0 - 0.5, num_cols - x1 - 0.5))
        ax.set_ylim((y0 - 0.5, num_rows - y1 - 0.5)[::-1])

        ax.set_xticks(np.arange(x0, num_cols - x1, 1))
        ax.xaxis.set_tick_params(labelsize=5)
        ax.set_yticks(np.arange(y0, num_rows - y1, 1))
        ax.yaxis.set_tick_params(labelsize=5)

        ax.imshow(vf, origin='lower', vmin=vmin, vmax=vmax)
        ax.set_aspect(1)

        scale = np.abs(vf).max()
        ax.set_title(f"{title}_{scale:.4f}", fontdict={'fontsize': 8, 'fontweight': 'medium'})
        figure.tight_layout()
        return title, figure

    def plot_trajectories(self, title, policy, num_trajectories=10, jitter_scale=1.):
        figure = plt.figure()
        ax = plt.gca()
        self._set_gridworld_frame(ax)
        trajectories = []

        for _ in range(num_trajectories):
            state_vec = self.reset()
            state = state_vec.argmax()
            trajectory = [self.unflatten_state(state_vec), ]

            for _ in range(99):
                a_distr = np.array(policy[state])
                action = np.random.choice(self.num_actions, p=a_distr / a_distr.sum(-1))
                state_vec, reward, done, info = self.step(action)
                state = state_vec.argmax()
                trajectory.append(self.unflatten_state(state_vec))
                if done:
                    break

            trajectories.append(np.array(trajectory))

        # trajectories_unflat = list(self.unflat_trajectories(trajectories))
        # unflatten_state(onehot_state, self.size, self.has_absorbing_state)
        # map(lambda traj: list(map(self._unflatten, traj)), trajectories)

        for traj in trajectories:
            trajj = traj + jitter_scale * np.random.rand(*traj.shape) / (2 * self.size)
            r, c = trajj.T
            ax.plot(c, r)

        return title, figure

    def enjoy_policy(self, policy, title):
        raise NotImplementedError
        gwp = emdp.gridworld.plotting.GridWorldPlotter(self)

        trajectories = []
        trajectory = []
        for _ in tqdm.trange(1):
            trajectory.clear()
            trajectory.append(self.reset())
            state = trajectory[-1]

            for _ in range(self.size * self.size * 100):
                a = policy(state)
                state, reward, done, info = self.step(a)
                trajectory.append(state)
            trajectories.append(trajectory)

        fig = plt.figure(figsize=(10, 4))
        pos = 121

        ax = fig.add_subplot(pos)
        ax.title.set_text(title)
        # ax.set_xlim(?, 0)  # decreasing time

        # trajectory
        gwp.plot_trajectories(ax, trajectories)
        gwp.plot_grid(ax)

        # ax = fig.add_subplot(pos + 1)
        # ax.set_ylim((-0.5, 2.5))
        gwp.plot_heatmap(ax, trajectories)

        gwp.plot_grid(ax)
        plt.show()
        self.reset()

    def seed(self, seed):
        self.set_seed(seed)

    def force_task(self, forced_task_idx):
        self.goal = self.goals[forced_task_idx]
        self.reward, self.terminal_matrix, self.transition = self.rebuild_mdp()
        super().reset()
        return self.current_state

    def render(self, mode='ansi'):
        # self.plot_s(self.current_state)
        room = self.numpy_room.copy()
        (p0, p1) = self.unflatten_state(self.current_state)
        room[self.goal[0]][self.goal[1]] = 'G'
        room[p0][p1] = '@'
        if mode == "ansi" or mode == "ascii":
            return room
        elif mode == "human":
            print(room)
        elif mode == "rgb_array":
            # Turn the ansi room into a rgb array
            room = np.array(list(map(lambda l: list(map(lambda c: self.ansi_to_rgb[c], l)), room)))
            room = nn_resample(room, (self.size * TILE_PIXELS, self.size * TILE_PIXELS))
            return room.astype(np.uint8)

    def observation(self):
        obs = super().observation()
        if isinstance(self.observation_space, gym.spaces.Dict) and hasattr(self, 'goal'):
            task_idx = np.eye(self.num_tasks)[self.goals.index(self.goal)]
            room = self.numpy_room.copy()
            (p0, p1) = self.unflatten_state(self.current_state)
            room[p0][p1] = '@'
            img_room = np.array(list(map(lambda l: list(map(lambda c: self.ansi_to_rgb[c], l)), room)))
            img_room = np.transpose(img_room, (2, 0, 1))
            return {
                'task': task_idx,
                'image': img_room,
            }
        else:
            return obs


# https://stackoverflow.com/questions/69728373/resize-1-channel-numpy-image-array-with-nearest-neighbour-interpolation
def nn_resample(img, shape):
    def per_axis(in_sz, out_sz):
        ratio = 0.5 * in_sz / out_sz
        return np.round(np.linspace(ratio - 0.5, in_sz - ratio - 0.5, num=out_sz)).astype(int)

    return img[per_axis(img.shape[0], shape[0])[:, None],
               per_axis(img.shape[1], shape[1])]

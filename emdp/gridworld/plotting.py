import numpy as np

# from .env import GridWorldMDP
import emdp.gridworld
from .helper_utilities import unflatten_state


class GridWorldPlotter:
    def __init__(self, mdp):
        self.mdp = mdp

    def _unflatten(self, onehot_state):
        return unflatten_state(onehot_state, self.size, self.has_absorbing_state)

    def plot_grid(self, ax):
        """
        Plots the skeleton of the grid world
        :param ax:
        :return:
        """
        for i in range(self.size + 1):
            ax.plot(np.arange(self.size + 1) - 0.5, np.ones(self.size + 1) * i - 0.5, color='k')

        for i in range(self.size + 1):
            ax.plot(np.ones(self.size + 1) * i - 0.5, np.arange(self.size + 1) - 0.5, color='k')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(False)

        return ax

    def plot_trajectories(self, ax, trajectories, dont_unflatten=False, jitter_scale=1):
        """
        Plots a individual trajectory paths with some jitter.
        :param ax: The axes to plot this on
        :param trajectories: a list of trajectories. Each trajectory is a list of states (numpy arrays)
                             These states should be obtained by using the mdp.step() operation. To prevent
                             this automatic conversion use `dont_unflatten`
        :param dont_unflatten: will not automatically unflatten the trajectories into (x,y) pairs.
                            (!) this assumes you have already unflattened them!
        :return:
        """

        if not dont_unflatten:
            trajectories_unflat = list(self.unflat_trajectories(trajectories))
        else:
            trajectories_unflat = trajectories

        for trajectory_unflattened in trajectories_unflat:
            x, y = list(zip(*trajectory_unflattened))
            x = np.array(x) + jitter_scale * np.random.rand(len(x)) / (2 * self.size)
            y = np.array(y) + jitter_scale * np.random.rand(len(x)) / (2 * self.size)
            ax.plot(x, y)

        return ax

    def plot_environment(self, ax, wall_locs=None, plot_grid=False):
        """
        Plots the environment  with walls.
        :param ax: The axes to plot this on
        :param wall_locs: Locations of the walls for plotting them in a different color..
        :return:
        """

        # plot states with background color white
        state_background = np.ones((self.size, self.size))

        # plot walls in lame way -- set them to some hand-engineered color
        wall_img = np.zeros((self.size, self.size, 4))
        if wall_locs is not None:
            for state in wall_locs:
                y_coord = state[0]
                x_coord = state[1]
                wall_img[y_coord, x_coord, 0] = 0.0  # R
                wall_img[y_coord, x_coord, 1] = 0.0  # G
                wall_img[y_coord, x_coord, 2] = 0.0  # B
                wall_img[y_coord, x_coord, 3] = 1.0  # alpha

        # render heatmap and overlay the walls image
        imshow_ax = ax.imshow(state_background, interpolation=None)
        imshow_ax = ax.imshow(wall_img, interpolation=None)
        ax.grid(False)

        # Switch on flag if you want to plot grid
        if plot_grid:
            for i in range(self.size + 1):
                ax.plot(np.arange(self.size + 1) - 0.5, np.ones(self.size + 1) * i - 0.5, color='k')

            for i in range(self.size + 1):
                ax.plot(np.ones(self.size + 1) * i - 0.5, np.arange(self.size + 1) - 0.5, color='k')
            ax.set_xlabel('x')
            ax.set_ylabel('y')

        return ax, imshow_ax

    def plot_heatmap(self, ax, trajectories, dont_unflatten=False, wall_locs=None):
        """
        Plots a state-visitation heatmap with walls.
        :param ax: The axes to plot this on.
        :param trajectories: a list of trajectories. Each trajectory is a list of states (numpy arrays)
                             These states should be obtained by using the mdp.step() operation. To prevent
                             this automatic conversion use `dont_unflatten`
        :param dont_unflatten: will not automatically unflatten the trajectories into (x,y) pairs.
                            (!) this assumes you have already unflattened them!
        :param wall_locs: Locations of the walls for plotting them in a different color..
        :return:
        """
        if not dont_unflatten:
            trajectories_unflat = list(self.unflat_trajectories(trajectories))
        else:
            trajectories_unflat = trajectories

        state_visitations = np.zeros((self.size, self.size))
        # plot actual state visitation heatmap
        for trajectory in trajectories_unflat:
            for state in trajectory:
                x_coord = state[0]
                y_coord = state[1]
                state_visitations[y_coord, x_coord] += 1.
        # plot walls in lame way -- set them to some hand-engineered color
        wall_img = np.zeros((self.size, self.size, 4))
        if wall_locs is not None:
            mid_visits = (np.max(state_visitations) - np.min(state_visitations)) / 2.
            for state in wall_locs:
                y_coord = state[0]
                x_coord = state[1]
                wall_img[y_coord, x_coord, 0] = 0.6  # R
                wall_img[y_coord, x_coord, 1] = 0.4  # G
                wall_img[y_coord, x_coord, 2] = 0.4  # B
                wall_img[y_coord, x_coord, 3] = 1.0  # alpha
        # render heatmap and overlay the walls image
        imshow_ax = ax.imshow(state_visitations, interpolation=None)
        imshow_ax = ax.imshow(wall_img, interpolation=None)
        ax.grid(False)
        return ax, imshow_ax

    def unflat_trajectories(self, trajectories):
        """
        Returns a generator where the trajectories have been unflattened.
        :param trajectories:
        :return:
        """
        return map(lambda traj: list(map(self._unflatten, traj)), trajectories)


def enjoy_policy(env, intrinsic_reward, policy):
    state = env.reset()
    cum_reward = 0
    cum_intr_reward = 0
    for _ in range(100):
        action = int(np.argmax(policy[state]))
        # env.render()

        print(["V", "^", ">", "<", "pickup", "dropoff"][action])

        new_state, reward, done, _ = env.step(action)

        cum_reward += reward
        cum_intr_reward += intrinsic_reward[state, action]

        if done:
            state = env.reset()
        else:
            state = new_state
    return cum_intr_reward, cum_reward

"""
Utilities to help build more complex grid worlds.
"""
import numpy as np

import emdp.actions
from .helper_utilities import build_simple_grid, flatten_state


class TransitionMatrixBuilder(object):
    """
    Builder object to build a transition matrix for a grid world
    """

    def __init__(self, grid_size, action_space, p_success):
        self.grid_size = grid_size
        self.action_space = action_space
        self.state_space = grid_size * grid_size
        self.grid_added = False
        self.P = build_simple_grid(size=self.grid_size, p_success=p_success)

    def add_wall_at(self, tuple_location):
        """
        Add a blockade at this position
        :param tuple_location: (x,y) location of the wall
        :return:
        """
        target_state = flatten_state(tuple_location, self.grid_size, self.state_space)
        target_state = target_state.argmax()
        # find all the ways to go to "target_state"
        # from_states contains states that can lead you to target_state by executing from_action
        from_states, from_actions = np.where(self.P[:, :, target_state] != 0)

        # get the transition probability distributions that go from s--> t via some action
        transition_probs_from = self.P[from_states, from_actions, :]
        # TODO: optimize this loop
        for i, from_state in enumerate(from_states):  # enumerate over states
            tmp = transition_probs_from[i, target_state]  # get the prob of transitioning
            transition_probs_from[i, target_state] = 0  # set it to zero
            transition_probs_from[i, from_state] += tmp  # add the transition prob to staying in the same place

        self.P[from_states, from_actions, :] = transition_probs_from

        # Get the probability of going to any state for all actions from target_state.
        transition_probs_from_wall = self.P[target_state, :, :]
        for i, probs_from_action in enumerate(transition_probs_from_wall):
            # Reset the probabilities.
            transition_probs_from_wall[i, :] = 0.0
            # Set the probability of going to the target state to be 1.0
            transition_probs_from_wall[i, target_state] = 1.0
        # Now set the probs of going to any state from target state as above (i.e only targets).
        self.P[target_state, :, :] = transition_probs_from_wall

        # renormalize and update transition matrix.
        normalization = self.P.sum(2)
        # normalization[normalization == 0] = 1
        normalization = 1 / normalization
        self.P = (self.P * np.repeat(normalization, self.P.shape[0]).reshape(*self.P.shape))

        assert np.allclose(self.P.sum(2), 1), 'Normalization did not occur correctly: {}'.format(self.P.sum(2))
        assert np.allclose(self.P[target_state, :, target_state], 1.0), 'All actions from wall should lead to wall!'

    def add_wall_between(self, start, end):
        """
        Adds a wall between the starting and ending location
        :param start: tuple (x,y) representing the starting position of the wall
        :param end: tuple (x,y) representing the ending position of the wall
        :return:
        """
        if not (start[0] == end[0] or start[1] == end[1]):
            raise ValueError('Walls can only be drawn in straight lines. '
                             'Therefore, at least one of the x or y between '
                             'the states should match.')

        if start[0] == end[0]:
            direction = 1
        else:
            direction = 0

        constant_idx = start[int(not direction)]
        start_idx = start[direction]
        end_idx = end[direction]

        if end_idx < start_idx:
            # flip start and end directions
            # to ensure we can still draw walls
            start_idx, end_idx = end_idx, start_idx

        for i in range(start_idx, end_idx + 1):
            my_location = [None, None]
            my_location[direction] = i
            my_location[int(not direction)] = constant_idx
            print(my_location)
            self.add_wall_at(tuple(my_location))


def create_reward_matrix(state_space, size, reward_spec, action_space=4):
    """
    Abstraction to create reward matrices.
    :param state_space: Size of the state space
    :param size: Size of the gird world (width)
    :param reward_spec: The reward specification
    :param action_space: The size of the action space
    :return:
    """
    R = np.zeros((state_space, action_space), dtype=np.float32)
    for (s0, a, reward_value) in reward_spec:
        s0 = flatten_state(s0, size, state_space).argmax()
        R[s0, a] = reward_value

    return R


"""
Simple builders for gridworlds
"""

# def build_simple_grid_world_with_terminal_states(reward_spec,
#                                                  size,
#                                                  p_success=1,
#                                                  gamma=0.99,
#                                                  seed=2017,
#                                                  start_state=0):
#     """
#     A simple size x size grid world where agents actions has a prob of p_success of executing correctly.
#     rewards are given by a dict where the indices and the x,y positions and the value is the magnitude of the reward.
#     Upon reaching a state with a reward, every action gives a reward. The episode then goes to an absorbing state and terminates.
#     :param reward_spec: Reward specification
#     :param size: Size of the gridworld (grid world will be size x size)
#     :param p_success: The probability the action is successful.
#     :param gamma: The discount factor.
#     :param seed: Seed for the GridWorldMDP object.
#     :param start_state: The index of the starding state.
#     :return:
#     """
#     P = build_simple_grid(size=size, terminal_states=reward_spec.keys(), p_success=p_success)
#     R = create_reward_matrix(P.shape[0], size, reward_spec, action_space=4)
#     p0 = np.zeros(P.shape[0])
#     p0[start_state] = 1
#
#     return GridWorldMDP(P, R, gamma, p0, terminal_states=reward_spec.keys(), size=size, seed=seed)
#
#
# def build_simple_grid_world_without_terminal_states(reward_spec,
#                                                     size,
#                                                     p_success=1,
#                                                     gamma=0.99,
#                                                     seed=2017,
#                                                     start_state=0):
#     """
#     A simple size x size grid world where agents actions has a prob of p_success of executing correctly.
#     rewards are given by a dict where the indices and the x,y positions and the value is the magnitude of the reward.
#     Upon reaching a state with a reward, every action gives a reward. The episode does not terminate.
#     :param reward_spec: Reward specification
#     :param size: Size of the gridworld (grid world will be size x size)
#     :param p_success: The probability the action is successful.
#     :param gamma: The discount factor.
#     :param seed: Seed for the GridWorldMDP object.
#     :param start_state: The index of the starting state.
#     :return:
#     """
#     P = build_simple_grid(size=size, terminal_states=[], p_success=p_success)
#     R = create_reward_matrix(P.shape[0], size, reward_spec, action_space=4)
#     p0 = np.zeros(P.shape[0])
#     p0[start_state] = 1
#
#

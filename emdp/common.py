import collections

import numpy as np

from . import utils
from .exceptions import InvalidActionError, EpisodeDoneError

MDP_ = collections.namedtuple("MDP", "P r discount s0")
MDPs_ = collections.namedtuple("MDP", "P rs discount s0")


class Env:
    """
    Abstract Environment wrapper.
    """

    def __init__(self, seed):
        """
        :param seed: A seed for the random number generator.
        """
        self.set_seed(seed)

    def set_seed(self, seed):
        self.rng = np.random.RandomState(seed)


class MDP(Env):
    def __init__(self, transition, reward, discount, initial_state, terminal_matrix, seed=1337, max_steps=100):
        """
        A simple MDP simulator.
        :param transition: The transition matrix of size |S|x|A|x|S|
        :param reward: The reward criterion |S|x|A|
        :param discount: the discount factor.
        :param initial_state: the distribution over starting states |S| (must sum to 1.)
        :param terminal_matrix: A list of integers which indicate terminal states, used to end episodes.
                                Note that in the transition matrix these
                                should be absorbing states to ensure calculations are correct.
        :param seed: the random seed for simulations.
        """
        super().__init__(seed)

        self.transition = transition
        self.reward = reward
        self.num_states, self.num_actions, _next_states = transition.shape

        assert np.allclose(transition.sum(axis=2), 1), 'Transition matrix does not seem to be a stochastic matrix (i.e. the sum over states for each action doesn not equal 1'
        assert self.num_states == transition.shape[2], '3rd Dimension of Transition Matrix is not of size |S|'
        assert self.num_actions == transition.shape[1], '2nd Dimension of Transition Matrix is not of size |A|'
        assert self.num_states == reward.shape[0], '1st Dimension of Reward Matrix is not of size |S|'
        assert self.num_states == initial_state.shape[0], 'Distribution over initial states is not over |S|'

        self.discount = discount
        self.initial_state = initial_state
        self.terminal_matrix = terminal_matrix
        self.current_state = None
        self.num_steps = None
        self.current_state_idx = None
        self.requires_reset = True
        self.max_steps = max_steps

    def mdp(self):
        return MDP_(self.transition, self.reward, self.discount, self.initial_state)

    def set_state(self, state_idx):
        self.current_state_idx = state_idx
        self.current_state = utils.convert_int_rep_to_onehot(state_idx, self.num_states)

    def reset(self):
        integer_representation = np.random.choice(np.arange(self.num_states), p=self.initial_state)
        self.set_state(integer_representation)
        self.requires_reset = False
        self.num_steps = 0
        return self.observation()

    def observation(self):
        return self.current_state

    def set_current_state_to(self, state):
        self.set_state(state)
        self.requires_reset = False
        return self.current_state

    def step(self, action):
        """
        :param action: An integer representing the action taken.
        :return:
        """
        if self.requires_reset:
            raise EpisodeDoneError('The episode has terminated. Use .reset() to restart the episode.')

        if np.issubdtype(type(action), np.integer):
            action = int(action)

        if action >= self.num_actions or not isinstance(action, int):
            raise InvalidActionError('Invalid action {}. It must be an integer between 0 and {}'.format(action, self.num_actions - 1))

        # we end from this episode onwards.
        # this check is done after entering terminal state
        # because we can only give the reward after leaving
        # a terminal state.

        # self.current_state.argmax()
        if self.terminal_matrix[self.current_state_idx, action] or self.num_steps == self.max_steps:
            self.requires_reset = True

        reward = self.reward[self.current_state_idx, action]

        if reward != 0 and not self.requires_reset:
            raise ValueError

        next_state_probs = self.transition[self.current_state_idx, action]

        sampled_next_state = self.rng.choice(np.arange(self.num_states), p=next_state_probs)
        self.set_state(sampled_next_state)
        self.num_steps += 1

        return self.observation(), reward, self.requires_reset, {'gamma': self.discount}

    def torch(self):
        import torch
        self.reward = torch.tensor(self.reward, dtype=torch.float)
        self.transition = torch.tensor(self.transition, dtype=torch.float)
        self.initial_state = torch.tensor(self.initial_state, dtype=torch.float)
        self.discount = torch.tensor(self.discount, dtype=torch.float)
        return self


class MultiTaskMDP(Env):
    def __init__(self, transition, rewards, discount, initial_state, terminal_matrix, seed=1337):
        super().__init__(seed)

        self.transition = transition
        self.rewards = rewards
        self.num_states = transition.shape[0]
        self.num_actions = reward.shape[1]

        assert np.allclose(transition.sum(axis=2), 1), 'Transition matrix does not seem to be a stochastic matrix (i.e. the sum over states for each action doesn not equal 1'
        assert self.num_states == transition.shape[2], '3rd Dimension of Transition Matrix is not of size |S|'
        assert self.num_actions == transition.shape[1], '2nd Dimension of Transition Matrix is not of size |A|'
        assert self.num_states == reward.shape[0], '1st Dimesnion of Reward Matrix is not of size |S|'
        assert self.num_states == initial_state.shape[0], 'Distribution over initial states is not over |S|'

        self.discount = discount
        self.initial_state = initial_state
        self.terminal_matrix = terminal_matrix
        self.current_state = None
        self.done = None
        self.reset()

    def reset(self):
        integer_representation = np.random.choice(np.arange(self.num_states), p=self.initial_state)
        self.current_state = utils.convert_int_rep_to_onehot(integer_representation, self.num_states)
        self.done = False
        return self.current_state

    def set_current_state_to(self, state):
        self.current_state = utils.convert_int_rep_to_onehot(state, self.num_states)
        self.done = False
        return self.current_state

    def step(self, action):
        """
        :param action: An integer representing the action taken.
        :return:
        """
        if self.done:
            raise EpisodeDoneError('The episode has terminated. Use .reset() to restart the episode.')
        if action >= self.num_actions or not isinstance(action, int):
            raise InvalidActionError('Invalid action {}. It must be an integer between 0 and {}'.format(action, self.num_actions - 1))

        # we end from this episode onwards.
        # this check is done after entering terminal state
        # because we can only give the reward after leaving
        # a terminal state.
        state = self.current_state.argmax()
        if self.current_state.argmax() in self.terminal_matrix[state]:
            self.done = True

        # get the vector representing the next state probabilities:
        current_state_idx = utils.convert_onehot_to_int(self.current_state)
        next_state_probs = self.P[current_state_idx, action]

        # sample the next state
        sampled_next_state = self.rng.choice(np.arange(self.num_states), p=next_state_probs)
        # observe the reward
        reward = self.reward[current_state_idx, action]

        self.current_state = utils.convert_int_rep_to_onehot(sampled_next_state, self.num_states)

        return self.current_state, reward, self.done, {'gamma': self.discount}

    def mdps(self):
        return MDPs_(self.transition, self.rewards, self.discount, self.initial_state)

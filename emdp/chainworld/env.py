import numpy as np

from .. import actions
from ..common import MDP


class ChainMDP(MDP):
    num_actions = 2

    def __init__(self, num_states, reward_spec, p_success=1, starting_distribution=None, terminal_states=(), discount=0.9):
        """
            A simple chain world with states and 2 actions.
            Actions can fail with probability 1-p_success
            (!) note that you probably want your terminal state to be separate from
            the state where the reward is obtained.

            Example of how to use:
            ```python

            # a 7 state MDP where the agent starts in the middle
            # at the two ends are absorbing states (given by terminal states)
            # if the agent reaches the state before the terminal state it gets a reward
            # if the agent is at the left of the world and it takes an action LEFT it gets a -1
            # otherwise it gets nothing
            # if the agent is at the right of the world and it takes an action RIGHT it gets a +1
            # otherwise it gets nothing
            build_chain_MDP(n_states=7, p_success=0.9, reward_spec=[(5, RIGHT, +1), (1, LEFT, -1)]
            starting_distribution=np.array([0,0,0,1,0,0,0]),
            terminal_states=[0, 6], gamma=0.9)
            ```
            :param num_states: the number of states in the chain world.
            :param p_success: the probability of successfully executing an action.
            :param reward_spec: a list of tuples which represent
            (location_of_reward, magnitude_of_reward)
            :param starting_distribution: a distribution over starting states.
            :param terminal_states: a list of integers representing the terminal states
            :param return_MDP: returns an MDP object, else will return the components to create one.
            :return:

            """
        if reward_spec is None:
            # By default the last state is rewarding
            reward_spec = [(num_states, actions.RIGHT, +1)]
        if starting_distribution is None:
            starting_distribution = np.zeros(num_states)
            starting_distribution[0] = 1.
        assert starting_distribution.sum() == 1.

        p_fail = 1 - p_success
        assert 1 >= p_success >= 0

        # building the transition matrix.
        P = np.zeros((num_states, self.num_actions, num_states))
        for s in range(num_states):
            if s in terminal_states:
                # whatever action we take from this state should end up in this state again
                P[s, :, s] = 1
            else:
                if s == 0:
                    # we are at the left edge of the grid.
                    # if we take the LEFT action it should be a no-op.
                    P[s, actions.LEFT, s - 1] = 0
                    P[s, actions.LEFT, s] = 1
                else:
                    # not at the left edge, fill in LEFT operation as usual
                    P[s, actions.LEFT, s - 1] = p_success  # successfully transition to the left
                    P[s, actions.LEFT, s] = p_fail

                if s == num_states - 1:
                    # we are at the right edge of the grid.
                    # if we take RIGHT action it should be a no-op
                    P[s, actions.RIGHT, s] = 1
                else:
                    # not at the right edge, fill in RIGHT operation as usual
                    P[s, actions.RIGHT, s + 1] = p_success  # successfully transition to the right
                    P[s, actions.RIGHT, s] = p_fail

        R = np.zeros((num_states, self.num_actions))
        for (reward_loc, action, reward_mag) in reward_spec:
            R[reward_loc, action] = reward_mag  # any action at this position leads to a reward.

        super(ChainMDP, self).__init__(P, R, discount, starting_distribution, terminal_states, seed=1337)

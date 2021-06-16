"""
A simple grid world environment
"""
from .helper_utilities import flatten_state, unflatten_state
from ..common import MDP


class GridWorldMDP(MDP):
    def __init__(self, transition, reward, discount, initial_state, terminal_states, size, seed=1337, convert_terminal_states_to_ints=False):
        """
        (!) if terminal_states is not empty then there will be an absorbing state. So
            the actual number of states will be size x size + 1
            if there is a terminal state, it should be the last one.
        :param transition: Transition matrix |S| x |A| x |S|
        :param reward: Transition matrix |S| x |A|
        :param discount: discount factor
        :param initial_state: initial starting distribution
        :param terminal_states: Must be a list of (x,y) tuples.  use skip_terminal_state_conversion if giving ints
        :param size: the size of the grid world (i.e there are size x size (+ 1)= |S| states)
        :param seed:
        :param validate_arguments:
        """
        if not convert_terminal_states_to_ints:
            terminal_states = list(map(lambda tupl: int(size * tupl[0] + tupl[1]), terminal_states))
        self.size = size
        self.human_state = (None, None)
        self.has_absorbing_state = len(terminal_states) > 0
        super().__init__(transition, reward, discount, initial_state, terminal_states, seed=seed)

    def reset(self):
        super().reset()
        self.human_state = self.unflatten_state(self.current_state)
        return self.current_state

    def flatten_state(self, state):
        """Flatten state (x,y) into a one hot vector"""
        return flatten_state(state, self.size, self.num_states)

    def unflatten_state(self, onehot):
        """Unflatten a one hot vector into a (x,y) pair"""
        return unflatten_state(onehot, self.size, self.has_absorbing_state)

    def step(self, action):
        state, reward, done, info = super().step(action)
        self.human_state = self.unflatten_state(self.current_state)
        return state, reward, done, info

    def set_current_state_to(self, tuple_state):
        return super().set_current_state_to(self.flatten_state(tuple_state).argmax())

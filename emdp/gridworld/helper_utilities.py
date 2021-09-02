import numpy as np

from .. import actions
from ..exceptions import InvalidActionError

n_actions = 4


def flatten_state(state, size, state_space):
    """Flatten state (r,c) into a one hot vector"""
    r, c = state
    idx = r * size + c
    one_hot = np.zeros(state_space)
    one_hot[idx] = 1
    return one_hot


def unflatten_state(onehot, size):
    """Unflatten a one hot vector into a (x,y) pair"""
    (s,), = np.argwhere(onehot)
    return state_to_xy(s, size)


def state_to_xy(s, size):
    r, c = divmod(s, size)
    return r, c


def get_state_after_executing_action(action, state, grid_size):
    """
    Gets the state after executing an action
    :param action:
    :param state:
    :param grid_size:
    :return:
    """
    ds = np.array([grid_size, 1])

    if check_can_take_action(action, state, grid_size):
        if action == actions.LEFT:
            t = state + ds.dot(actions.LEFT_rc)
            assert (state - 1) == t
        elif action == actions.RIGHT:
            t = state + ds.dot(actions.RIGHT_rc)
            assert (state + 1) == t
        elif action == actions.UP:
            t = state + ds.dot(actions.UP_rc)
            assert t == (state - grid_size)
        elif action == actions.DOWN:
            t = state + ds.dot(actions.DOWN_rc)
            assert t == (state + grid_size)
        else:
            raise ValueError
        return t
    else:
        # cant execute action, stay in the same place.
        return state


def check_can_take_action(action, state, grid_size):
    """
    checks if you can take an action in a state.
    :param action:
    :param state:
    :param grid_size:
    :return:
    """
    LAST_ROW = list(range(grid_size * (grid_size - 1), grid_size * grid_size))
    FIRST_ROW = list(range(0, grid_size))
    LEFT_EDGE = list(range(0, grid_size * grid_size, grid_size))
    RIGHT_EDGE = list(range(grid_size - 1, grid_size * grid_size, grid_size))

    if action == actions.DOWN:
        if state in LAST_ROW:
            return False
    elif action == actions.RIGHT:
        if state in RIGHT_EDGE:
            return False
    elif action == actions.UP:
        if state in FIRST_ROW:
            return False
    elif action == actions.LEFT:
        if state in LEFT_EDGE:
            return False
    else:
        raise InvalidActionError('Cannot take action {} in a grid world of size {}x{}'.format(action, grid_size, grid_size))

    return True


def get_possible_actions(state, grid_size):
    """
    Gets all possible actions at a given state.
    :param state:
    :param grid_size:
    :return:
    """
    LAST_ROW = list(range(grid_size * (grid_size - 1), grid_size * grid_size))
    FIRST_ROW = list(range(0, grid_size))
    LEFT_EDGE = list(range(0, grid_size * grid_size, grid_size))
    RIGHT_EDGE = list(range(grid_size - 1, grid_size * grid_size, grid_size))

    available_actions = actions.ACTIONS_ORDER.copy()
    if state in LAST_ROW:
        available_actions.remove(actions.DOWN)
    if state in FIRST_ROW:
        available_actions.remove(actions.UP)
    if state in RIGHT_EDGE:
        available_actions.remove(actions.RIGHT)
    if state in LEFT_EDGE:
        available_actions.remove(actions.LEFT)
    return available_actions


# def flatten_state(state, n_states, grid_size):
#     """Flatten state (x,y) into a one hot vector"""
#     idx =
#     one_hot = np.zeros(n_states)
#     one_hot[idx] = 1
#     return one_hot

def build_simple_grid(size=5, p_success=1):
    """
    Builds a simple grid where an agent can move LEFT, RIGHT, UP or DOWN
    and actions success with probability p_success.
    A terminal state is added if len(terminal_states) > 0 and will return matrix of
    size (|S|+1)x|A|x(|S|+1)

    Moving into walls does nothing.
    :param size: size of the grid world
    :param p_success: the probabilty that an action will be successful.
    :return:
    """
    p_fail = 1 - p_success

    n_states = size * size

    # this helper function creates the state transition list for
    # taking an action in a state
    def create_state_list_for_action(state_idx, action):
        transition_probs = np.zeros(n_states)

        if action in actions.ACTIONS_ORDER:
            # valid action, now see if we can actually execute this action
            # in this state:
            # TODO: distinguish between capability of slipping and taking wrong action vs failing to execute action.
            if check_can_take_action(action, state_idx, size):
                # yes we can
                possible_actions = get_possible_actions(state_idx, size)
                if action in possible_actions:
                    transition_probs[get_state_after_executing_action(action, state_idx, size)] = p_success
                    possible_actions.remove(action)
                for other_action in possible_actions:
                    transition_probs[get_state_after_executing_action(other_action, state_idx, size)] = p_fail / len(possible_actions)

            else:
                possible_actions = get_possible_actions(state_idx, size)
                transition_probs[state_idx] = p_success  # cant take action, stay in same place
                for other_action in possible_actions:
                    transition_probs[get_state_after_executing_action(other_action, state_idx, size)] = p_fail / len(possible_actions)

        else:
            raise InvalidActionError('Invalid action {} in the 2D gridworld'.format(action))
        return transition_probs

    P = np.zeros((n_states, n_actions, n_states))
    for s in range(n_states):
        for a in range(n_actions):
            P[s, a, :] = create_state_list_for_action(s, a)
    return P


def add_walls():
    pass

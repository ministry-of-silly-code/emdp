"""Utilities to help load gridworlds from a text file.
"""
from .helper_utilities import flatten_state
from .builder_tools import (TransitionMatrixBuilder, create_reward_matrix)
# from . import GridWorldMDP

def get_char_matrix(raw_file):
    """
    :param raw_file: Either a python file object (open)
        or a list of strings containing the lines.
    """
    return [[c for c in line.strip('\n')] for line in raw_file]


def build_gridworld_from_char_matrix(
  char_matrix,
  p_success=1,
  seed=2017,
  gamma=1,
  skip_checks=False,
  terminal_states=None,
  transition_matrix_builder_cls=TransitionMatrixBuilder):
    """
    A parser to build a gridworld from a text file.
    Each grid has ONE start and goal location.
    A reward of +1 is positioned at the goal location.
    :param char_matrix: Matrix of characters.
    :param p_success: Probability that the action is successful.
    :param seed: The seed for the GridWorldMDP object.
    :param skip_checks: Skips assertion checks.
    :transition_matrix_builder_cls: The transition matrix builder to use.
    :return:
    """
    grid_size = len(char_matrix[0])

    if not skip_checks:
        assert(len(char_matrix) == grid_size), 'Mismatch in the columns.'
        for row in char_matrix:
            assert(len(row) == grid_size), 'Mismatch in the rows.'
    # ...
    wall_locs = []
    start_loc = None
    goal_loc = None
    for r in range(grid_size):
        for c in range(grid_size):
            char = char_matrix[r][c]
            if char == '#':
                wall_locs.append((r, c))
            elif char == 's':
                assert start_loc is None, 'Start loc was overwritten!'
                start_loc = (r, c)
            elif char == 'g':
                assert goal_loc is None, 'Goal loc was overwritten!'
                goal_loc = (r, c)
            elif char != ' ':
                raise ValueError('Unknown character {} in grid.'.format(char))
    # Attempt to make the desired gridworld.
    reward_spec = {(goal_loc[0], goal_loc[1]): +1}


    tmb = transition_matrix_builder_cls(grid_size,  has_terminal_state=True)
    tmb.add_grid(terminal_states=reward_spec.keys(), p_success=p_success)
    for (r, c) in wall_locs:
        tmb.add_wall_at((r, c))
    P = tmb.P


    R = create_reward_matrix(P.shape[0], grid_size, reward_spec, action_space=4)
    p0 = flatten_state(start_loc, grid_size, R.shape[0])

    if terminal_states is None:
        terminal_states = reward_spec.keys()

    gw = GridWorldMDP(P, R, gamma, p0, terminal_states=terminal_states, size=grid_size, seed=seed)
    return gw, wall_locs

def ascii_to_walls(char_matrix):
    """
    A parser to build a gridworld from a text file.
    Each grid has ONE start and goal location.
    A reward of +1 is positioned at the goal location.
    :param char_matrix: Matrix of characters.
    :param p_success: Probability that the action is successful.
    :param seed: The seed for the GridWorldMDP object.
    :param skip_checks: Skips assertion checks.
    :transition_matrix_builder_cls: The transition matrix builder to use.
    :return:
    """
    grid_size = len(char_matrix[0])

    assert(len(char_matrix) == grid_size), 'Mismatch in the columns.'
    for row in char_matrix:
        assert(len(row) == grid_size), 'Mismatch in the rows.'
    # ...
    wall_locs = []
    empty = []

    for r in range(grid_size):
        for c in range(grid_size):
            char = char_matrix[r][c]
            if char == '#':
                wall_locs.append((r, c))
            elif char == ' ':
                empty.append((r, c))
            else:
                raise ValueError('Unknown character {} in grid.'.format(char))
    # Attempt to make the desired gridworld.
    return wall_locs, empty

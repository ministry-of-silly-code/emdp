import random

import numpy as np


import emdp.gridworld


class HierarchicalEnvironment(emdp.gridworld.GridWorldMDP):
    def __init__(self, mdp, options, option_termination, use_primitives=True):
        super().__init__(initial_states=mdp.initial_states, goals=mdp.goals, seed=mdp.initial_seed)
        self.num_primitive_actions = self.num_actions
        self.use_primitives = use_primitives
        if use_primitives:
            self.num_actions = self.num_actions + len(options)
        else:
            self.num_actions = len(options)
        self.options = options
        self.option_termination = option_termination
        self.last_state = None

    def is_option(self, action):
        return action >= self.num_primitive_actions

    def step(self, action):
        assert isinstance(action, int)
        if self.use_primitives:
            action += self.num_primitive_actions

        if action >= self.num_primitive_actions:
            option_idx = action - self.num_primitive_actions
            state = self.last_state
            terminal = False
            total_reward = 0.
            # TODO:
            while not terminal:
                action_distribution = self.options[option_idx, state]
                actions = np.arange(len(action_distribution))
                action_distribution = np.asarray(action_distribution)
                primitive_action = np.random.choice(actions, p=action_distribution)

                state_vec, reward, terminal, info = super().step(int(primitive_action))
                state = state_vec.argmax()
                total_reward += reward

                if random.random() < self.option_termination:
                    break

            # TODO: should this be discounted?
            reward = total_reward
        else:
            state_vec, reward, terminal, info = super().step(action)
            state = state_vec.argmax()

        self.last_state = state
        return state_vec, reward, terminal, info

    def reset(self):
        s = super().reset()
        self.last_state = s.argmax()
        return s

import random

import numpy as np

import emdp


class HierarchicalEnvironment(emdp.MDP):
    def __init__(self, mdp, options, option_termination):
        super().__init__(mdp.transition, mdp.reward, mdp.discount, mdp.initial_state, mdp.terminal_states)
        self.nr_primitive_actions = self.num_actions
        self.num_actions = self.num_actions + len(options)
        self.options = options
        self.option_termination = option_termination
        self.last_state = None

    def is_option(self, action):
        return action >= self.nr_primitive_actions

    def step(self, action):
        assert isinstance(action, int)

        if action >= self.nr_primitive_actions:
            option_idx = action - self.nr_primitive_actions
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
                    terminal = True

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

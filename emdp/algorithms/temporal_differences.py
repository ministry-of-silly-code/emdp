import random

import numpy as np
import tqdm


def sarsa(model, alpha=0.5, epsilon=0.1, maxiter=100, maxeps=1000):
    """
    Solves the supplied environment using SARSA.

    Parameters
    ----------
    model : python object
        Holds information about the environment to solve
        such as the reward structure and the transition dynamics.

    alpha : float
        Algorithm learning rate. Defaults to 0.5.

    epsilon : float
         Probability that a random action is selected. epsilon must be
         in the interval [0,1] where 0 means that the action is selected
         in a completely greedy manner and 1 means the action is always
         selected randomly.

    maxiter : int
        The maximum number of iterations to perform per episode.
        Defaults to 100.

    maxeps : int
        The number of episodes to run SARSA for.
        Defaults to 1000.

    Returns
    -------
    q : numpy array of shape (N, 1)
        The state-action value for the environment where N is the
        total number of states

    pi : numpy array of shape (N, 1)
        Optimal policy for the environment where N is the total
        number of states.

    state_counts : numpy array of shape (N, 1)
        Counts of the number of times each state is visited
    """
    # initialize the state-action value function and the state counts
    Q = np.zeros((model.num_states, model.num_actions))
    state_counts = np.zeros((model.num_states, 1))

    for i in range(maxeps):

        if np.mod(i, 1000) == 0:
            print("Running episode %i." % i)

        # for each new episode, start at the given start state
        state = int(model.start_state_seq)
        # sample first e-greedy action
        action = sample_action(Q, state, model.num_actions, epsilon)

        for j in range(maxiter):
            # initialize p and r
            p, r = 0, np.random.random()
            # sample the next state according to the action and the
            # probability of the transition
            for next_state in range(model.num_states):
                p += model.P[state, next_state, action]
                if r <= p:
                    break
            # epsilon-greedy action selection
            next_action = sample_action(Q, next_state, model.num_actions, epsilon)
            # Calculate the temporal difference and update Q function
            Q[state, action] += alpha * (model.R[state] + model.gamma * Q[next_state, next_action] - Q[state, action])
            # End episode is state is a terminal state

            if np.any(state == model.goal_states_seq):
                break

            # count the state visits
            state_counts[state] += 1

            # store the previous state and action
            state = next_state
            action = next_action

    # determine the q function and policy
    q = np.max(Q, axis=1).reshape(-1, 1)
    pi = np.argmax(Q, axis=1).reshape(-1, 1)

    return q, pi, state_counts


def qlearning(env, alpha=0.5, epsilon=0.1, max_samples=1000):
    q = 0.00001 * np.random.rand(env.num_states, env.num_actions)
    qmax = np.max(q, axis=1)
    policy = np.argmax(q, axis=1)

    _ = env.reset()
    state = env.current_state_idx

    for _ in tqdm.trange(max_samples, desc="q-learning"):
        if random.random() < epsilon:
            action = random.randrange(0, env.num_actions)
        else:
            action = policy[state]

        _, reward, done, _ = env.step(int(action))
        next_state = env.current_state_idx

        future_value = qmax[next_state]

        discounted_future_value = env.discount * future_value
        old_q = q[state, action]
        delta_Q = reward + discounted_future_value - old_q
        new_q = old_q + alpha * delta_Q

        q[state, action] = new_q

        if new_q > qmax[state]:
            qmax[state] = new_q
            policy[state] = action

        if done:
            _ = env.reset()
            state = env.current_state_idx
        else:
            state = next_state

    return q

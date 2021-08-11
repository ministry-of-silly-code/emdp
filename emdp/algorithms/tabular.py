import jax.numpy as np


def policies_evaluation(mdp, policies):
    ppi = np.einsum('ast,nsa->nst', mdp.P, policies)
    rpi = np.einsum('nsa,nsa->ns', mdp.rs, policies)
    vf = batched_value_function(mdp, ppi, rpi)
    return vf


def policy_evaluation(mdp, policy):
    assert np.all(policy.sum(1) == 1)
    ppi = np.einsum('ast,sa->st', mdp.P, policy)
    rpi = np.einsum('sa,sa->s', mdp.r, policy)
    vf = value_function(mdp, ppi, rpi)
    return vf


def expected_return(mdp, policy):
    vf = policies_evaluation(mdp, policy)
    return mdp.s0 @ vf


def solve_mdp(mdp):
    num_iters = 50
    n_actions, n_states, _ = mdp.P.shape
    qf = np.zeros((n_states, n_actions))
    for _ in range(num_iters):
        qf = mdp.R + mdp.discount * np.einsum('ast,t->sa', mdp.P, np.max(qf, axis=1))
    vf = np.max(qf, axis=1)
    pi = np.argmax(qf, axis=1)
    return pi


def value_function(mdp, p_pi, r_pi):
    num_states = mdp.P.shape[-1]
    I = np.eye(num_states)
    vf = np.linalg.solve(I - mdp.discount * p_pi, r_pi)
    return vf


def batched_value_function(mdp, p_pis, r_pis):
    I = np.eye(mdp.P.shape[-1])
    vfs = []
    for r_pi, p_pi in zip(r_pis, p_pis):
        vf = np.linalg.solve(I - mdp.discount * p_pi, r_pi)
        vfs.append(vf)
    vfs = np.stack(vfs, axis=0)
    return vfs
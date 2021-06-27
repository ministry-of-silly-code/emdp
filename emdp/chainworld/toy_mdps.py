def dadashi_fig2d():
    """ Figure 2 d) of
    ''The Value Function Polytope in Reinforcement Learning''
    by Dadashi et al. (2019) https://arxiv.org/abs/1901.11524

    Returns:
        tuple (P, R, gamma) where the first element is a tensor of shape
        (A x S x S), the second element 'R' has shape (S x A) and the
        last element is the scalar (float) discount factor.
    """
    P = np.array([[[0.7, 0.3], [0.2, 0.8]], [[0.99, 0.01], [0.99, 0.01]]])
    R = np.array(([[-0.45, -0.1], [0.5, 0.5]]))
    return MDP(P, R, 0.9)

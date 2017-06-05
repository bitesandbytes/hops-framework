"""
Class for a random deterministic policy for continuous action-space MDPs
Using a given state, a policy, mu(s) = a is obtained using mu(s) = A*phi(s)
where A.shape = [action space dimension, state space dimension]
"""

import numpy as np


class RandomDeterministicPolicy(object):
    def __init__(self, s_dim, a_dim):
        self.transform_matrix = np.random.rand(a_dim, s_dim)

    def step(state_vec):
        assert state_vec.shape[0] == self.transform_matrix.shape[
            1], 'RandomDeterministicPolicy::step() \n State dimension mismatch'
        return self.transform_matrix * state_vec

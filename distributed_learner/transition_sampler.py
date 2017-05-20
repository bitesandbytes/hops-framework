"""
Class to sample transitions
* Similar to Replay Buffer used in DQNs, but doesn't have concept of time
* Stores (S,A,R,S') as separate matrices s_buf, a_buf, r_buf, sp_buf
    - fast retrieval
    - slow insert
"""

import random
import numpy as np

class TransitionSampler(object):

    def __init__(self, rng_seed=10001):
        self.s_buf = np.array([])
        self.a_buf = np.array([])
        self.r_buf = np.array([])
        self.sp_buf = np.array([])
        random.seed(random_seed)

    # transition_array = [(s,a,r,s')]
    def add_transitions(self, transition_array):
        extra_s = [_[0] for _ in transition_array]
        extra_a = [_[1] for _ in transition_array]
        extra_r = [_[2] for _ in transition_array]
        extra_sp = [_[3] for _ in transition_array]

        np.append(self.s_buf, extra_s, axis=0)
        np.append(self.a_buf, extra_a, axis=0)
        np.append(self.r_buf, extra_r, axis=0)
        np.append(self.sp_buf, extra_sp, axis=0)

    # returns ([s], [a], [r], [s'])
    def sample_batch(self, batch_size):
        idxs = np.random.choice(range(0,self.s_buf.shape[0]), batch_size)
        s_batch = self.s_buf[idxs, :]
        a_batch = self.a_buf[idxs, :]
        r_batch = self.r_buf[idxs, :]
        sp_batch = self.sp_buf[idxs, :]

        return s_batch, a_batch, r_batch, sp_batch

    # Drop all transitions
    def clear(self):
        self.s_buf[:] = []
        self.a_buf[:] = []
        self.r_buf[:] = []
        self.sp_buf[:] = []

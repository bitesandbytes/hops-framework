import logging

import keras.backend as K
# from heraspy import *
import numpy as np
import tensorflow as tf
from keras.layers import Dense, Input, merge
from keras.models import Model
from keras.optimizers import Adam


# Learns an embedding function f:SxG->Z
# |S|~20, |G|~20
# Try 20-dim Z-space for embedding
class EmbeddingLearner(object):
    # inp_size = Nx(S+G)
    # emd_size = NxZ
    def __init__(self, args):
        #self.sess = args['sess']
        self.sess = tf.Session()
        self.state_size = args['state_size']
        self.goal_size = args['goal_size']
        self.embedding_size = args['emb_size']
        self.batch_size = args['batch_size']
        self.learning_rate = args['learning_rate']
        # self.idx = args['thread_idx']

        # init keras network
        # Build encoder from here
        S = Input(shape=(self.state_size,), name='state')
        G = Input(shape=(self.goal_size,), name='goal')
        S0 = Dense(self.state_size * 5, activation='relu')(S)
        G0 = Dense(self.goal_size * 5, activation='relu')(G)
        # C = merge.Concatenate([S0, G0])#, axis=1)
        C = merge([S0, G0], mode='concat')
        emb_layer = Dense(self.embedding_size, activation='relu')(C)
        # Build decoder from here
        S1 = Dense(self.state_size * 5, activation='relu')(emb_layer)
        G1 = Dense(self.goal_size * 5, activation='relu')(emb_layer)
        S2 = Dense(self.state_size, activation='relu')(S1)
        G2 = Dense(self.goal_size, activation='relu')(G1)
        # combine to form autoencoder
        self.autoencoder = Model(input=[S, G], output=[S2, G2])
        adam = Adam(lr=self.learning_rate)
        self.autoencoder.compile(loss='mse', optimizer=adam)

        self.encoder = Model(input=[S, G], output=emb_layer)

        K.set_session(self.sess)

        self.sess.run(tf.global_variables_initializer())

    # states=NxS, goals=NxG
    def fit(self, states, goals, val_ratio=0.05):
        logging.getLogger("learner").info("EMB fitting")
        # Prepare heras callback
        # herasCallback = HeraCallback('embedding-learner-'+str(self.idx), 'localhost', 9990+self.idx)

        # Split data into val and train dataset
        indices = np.random.permutation(states.shape[0])
        val_idx, train_idx = indices[:int(np.floor(states.shape[0] * val_ratio))], \
                             indices[int(np.floor(states.shape[0] * val_ratio)):]
        val_states, train_states = states[val_idx, :], states[train_idx, :]
        val_goals, train_goals = goals[val_idx, :], goals[train_idx, :]
        self.autoencoder.fit([train_states, train_goals], [train_states, train_goals],
                             epochs=50,
                             batch_size=self.batch_size,
                             shuffle=True,
                             validation_data=([val_states, val_goals], [val_states, val_goals]))
        # callbacks=[herasCallback])

    # states, goals = NxS, NxG
    # returns embed_matrix = NxZ
    def embed(self, states, goals):
        logging.getLogger(name="learner").info("embedding")
        return self.encoder.predict([states, goals])

import tensorflow as tf
from keras.models import Sequential, Model
from keras.engine.training import collect_trainable_weights
from keras.layers import Dense, Flatten, Input, merge, Lambda
from keras.optimizers import Adam
from heraspy import *
import matplotlib.pyplot as plt

# Learns an embedding function f:SxG->Z
# |S|<=20, |G|<= 20
# Try 10-dim Z-space for embedding
class EmbeddingLearner(object):
    # inp_dim = (S+G)
    # emd_dim = Z
    def __init__(self, idx, sess, s_dim, g_dim, emb_dim=10, learning_rate=0.001, batch_size=256):
        self.state_dim = s_dim
        self.goal_dim = g_dim
        self.embedding_dims = emb_dim
        self.batch_size=batch_size
        self.learning_rate=learning_rate
        self.idx = idx

        # init keras network
        # Build encoder from here
        S = Input(shape=(self.state_dim,))
        G = Input(shape=(self.goal_dim, ))
        S0 = Dense(self.state_dim*2, activation='relu')(S)
        G0 = Dense(self.goal_dim*2, actiation='relu')(G)
        C = merge([S0, G0], mode='concat')
        mid_layer = Dense(self.state_dim+self.goal_dim, activation='relu')(C)
        # Build decoder from here
        S1 = Dense(self.state_dim*2, activation='relu')(mid_layer)
        G1 = Dense(self.goal_dim*2, activation='relu')(mid_layer)
        S2 = Dense(self.state_dim, activation='relu')(S1)
        G2 = Dense(self.goal_dim, activation='relu')(G1)
        # combine to form autoencoder
        autoencoder = Model(input=[S, G], output=[S2, G2])
        adam = Adam(lr=self.learning_rate)
        autoencoder.compile(loss='mse', optimizer='adam')

        encoder = Model(input=[S,G], output=mid_layer)

        # set session
        self.session = sess
        K.set_session(sess)

    # states=NxS, goals=NxG
    def fit(states, goals, val_ratio=0.05, batch_size=256):
        # Prepare heras callback
        herasCallback = HeraCallback('embedding-learner-'+str(self.idx), 'localhost', 9990+self.idx)

        # Split data into val and train dataset
        indices = np.random.permutation(state_goal_matrix.shape[0])
        val_idx, train_idx = indices[:np.floor(n_pts*val_ratio)], indices[np.floor(n_pts*val_ratio):]
        val_states, train_states = states[val_idx,:], states[train_idx,:]
        val_goals, train_goals = goals[val_idx,:], goals[train_idx, :]
        self.autoencoder.fit([train_states,train_goals], [train_states, train_goals],
                             epochs=50,
                             batch_size=self.batch_size,
                             shuffle=True,
                             validation_data = ([val_states, val_goals], [val_states, val_goals]),
                             callbacks=[herasCallback])


    # states, goals = NxS, NxG
    # returns embed_matrix = NxZ
    def embed(states, goals):
        return self.encoder.predict([states, goals])

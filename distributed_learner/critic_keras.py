"""
Adapted from github.com/yanpanlau
"""

import logging

import keras.backend as K
import tensorflow as tf
from keras.layers import Dense, Input, merge
from keras.models import Model
from keras.optimizers import Adam

HIDDEN1_UNITS = 100
HIDDEN2_UNITS = 200


class CriticNetwork(object):
    def __init__(self, args):
        #self.sess = tf.Session()
        # self.sess = args['sess']
        self.sess = K.get_session()
        self.network_config = args['critic_network_config']  # xlayer_x_size, xlayer_x_type keys
        self.batch_size = args['batch_size']
        self.target_update_rate = args['target_update_rate']
        self.learning_rate = args['learning_rate']

        #K.set_session(self.sess)

        # Now create the model
        state_size, action_size = args['state_size'], args['action_size']

        # Now create the model
        self.model, self.action, self.state = self.create_critic_network(state_size, action_size)
        self.target_model, self.target_action, self.target_state = self.create_critic_network(state_size, action_size)
        self.action_grads = tf.gradients(self.model.output, self.action)  # GRADIENTS for policy update
        #self.sess.run(tf.initialize_all_variables())
        self.sess.run(tf.global_variables_initializer())

    def gradients(self, states, actions):
        logging.getLogger("learner").info("getting critic gradients")
        return self.sess.run(self.action_grads, feed_dict={
            self.state: states,
            self.action: actions
        })[0]

    def target_train(self):
        logging.getLogger("learner").info("critic target update")
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
        for i in xrange(len(critic_weights)):
            critic_target_weights[i] = self.target_update_rate * critic_weights[i] + (1 - self.target_update_rate) * \
                                                                                     critic_target_weights[i]
        self.target_model.set_weights(critic_target_weights)

    def create_critic_network(self, state_size, action_dim):
        logging.getLogger("learner").info("building critic")
        S = Input(shape=[state_size, ])
        A = Input(shape=[action_dim, ], name='action2')
        # HIDDEN1_UNITS, relu
        w1 = Dense(self.network_config['slayer1_size'], activation=self.network_config['slayer1_type'])(S)
        # HIDDEN2_UNITS, linear
        a1 = Dense(self.network_config['alayer_size'], activation=self.network_config['alayer_type'])(A)
        # HIDDEN2_UNITS, linear
        h1 = Dense(self.network_config['slayer2_size'], activation=self.network_config['slayer2_type'])(w1)
        h2 = merge([h1, a1], mode='mul')
        # HIDDEN2_UNITS, linear
        h3 = Dense(self.network_config['prefinal_layer_size'],
                   activation=self.network_config['prefinal_layer_type'])(h2)
        V = Dense(1, activation='linear')(h3)
        model = Model(input=[S, A], output=V)
        adam = Adam(lr=self.learning_rate)
        model.compile(loss='mse', optimizer=adam)
        return model, A, S

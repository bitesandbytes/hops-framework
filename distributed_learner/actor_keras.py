"""
Adapted from github.com/yanpanlau
"""

import logging

import keras.backend as K
import tensorflow as tf
from keras.initializers import TruncatedNormal
from keras.layers import Dense, Input
from keras.models import Model


class ActorNetwork(object):
    def __init__(self, args):
        self.sess = K.get_session()
        self.batch_size = args['batch_size']
        self.target_update_rate = args['target_update_rate']
        self.learning_rate = args['learning_rate']
        self.network_config = args['actor_network_config']  # hlayer_x_size, hlayer_x_type keys

        # K.set_session(self.sess)

        # Now create the model
        state_size, action_size = args['state_size'], args['action_size']

        self.model, self.weights, self.state = self.create_actor_network(state_size, action_size)
        self.target_model, self.target_weights, self.target_state = self.create_actor_network(state_size, action_size)
        self.action_gradient = tf.placeholder(tf.float32, [None, action_size])
        self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient)
        grads = zip(self.params_grad, self.weights)
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(grads)
        #self.sess.run(tf.initialize_all_variables())
        self.sess.run(tf.global_variables_initializer())

    def train(self, states, action_grads):
        logging.getLogger("learner").info("training actor")
        self.sess.run(self.optimize, feed_dict={
            self.state: states,
            self.action_gradient: action_grads
        })

    def target_train(self):
        logging.getLogger("learner").info("actor target update")
        cur_weights, target_weights = self.model.get_weights(), self.target_model.get_weights()
        for i in xrange(len(target_weights)):
            target_weights[i] = (1 - self.target_update_rate) * target_weights[i] + self.target_update_rate * \
                                                                                    cur_weights[i]
        self.target_model.set_weights(target_weights)

    def create_actor_network(self, state_size, action_dim):
        logging.getLogger("learner").info("building actor")
        S = Input(shape=[state_size, ], name='actor_input')
        # HIDDEN1_UNITS=100, relu
        h0 = Dense(self.network_config['hlayer_1_size'], activation=self.network_config['hlayer_1_type'],
                   name='actor_h0')(S)
        # HIDDEN2_UNITS=200, relu
        h1 = Dense(self.network_config['hlayer_2_size'], activation=self.network_config['hlayer_2_type'],
                   name='actor_h1')(h0)
        A = Dense(action_dim, activation='tanh', init=TruncatedNormal(stddev=0.5), name='actor_A')(h1)
        model = Model(input=S, output=A)
        return model, model.trainable_weights, S

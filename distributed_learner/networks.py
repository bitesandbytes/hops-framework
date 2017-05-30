import tensorflow as tf
from keras.initializers import TruncatedNormal
from keras.layers import Input, Dense, merge
from keras.models import Model


def create_actor(state_size, action_size, learning_rate):
    with tf.device("/cpu:0"):
        S = Input(shape=[state_size, ], name='actor_input')
        # HIDDEN1_UNITS=100, relu
        h0 = Dense(100, activation='relu', name='actor_h0')(
            S)
        # HIDDEN2_UNITS=200, relu
        h1 = Dense(200, activation='relu', name='actor_h1')(
            h0)
        A = Dense(action_size, activation='tanh', init=TruncatedNormal(stddev=0.5), name='actor_A')(h1)
        model = Model(input=S, output=A)
        action_gradient = tf.placeholder(tf.float32, [None, action_size])
        params_grad = tf.gradients(model.output, model.trainable_weights, -action_gradient)

    return model, action_gradient, params_grad


def create_critic(state_size, action_size, learning_rate):
    with tf.device("/cpu:0"):
        S = Input(shape=[state_size, ])
        A = Input(shape=[action_size, ], name='action2')
        # HIDDEN1_UNITS, relu
        w1 = Dense(100, activation='relu')(S)
        # HIDDEN2_UNITS, linear
        a1 = Dense(200, activation='linear')(A)
        # HIDDEN2_UNITS, linear
        h1 = Dense(200, activation='linear')(w1)
        h2 = merge([h1, a1], mode='mul')
        # HIDDEN2_UNITS, linear
        h3 = Dense(200, activation='linear')(h2)
        q = Dense(1, activation='linear')(h3)
        model = Model(input=[S, A], output=q)
        action_grads = tf.gradients(model.output, A)

    return model, action_grads


def create_emb_learner(state_size, goal_size, emb_size):
    with tf.device("/cpu:0"):
        S = Input(shape=(state_size,), name='state')
        G = Input(shape=(goal_size,), name='goal')
        S0 = Dense(state_size * 5, activation='relu')(S)
        G0 = Dense(goal_size * 5, activation='relu')(G)
        # C = merge.Concatenate([S0, G0])#, axis=1)
        C = merge([S0, G0], mode='concat')
        emb_layer = Dense(emb_size, activation='relu')(C)
        # Build decoder from here
        S1 = Dense(state_size * 5, activation='relu')(emb_layer)
        G1 = Dense(goal_size * 5, activation='relu')(emb_layer)
        S2 = Dense(state_size, activation='relu')(S1)
        G2 = Dense(goal_size, activation='relu')(G1)

        autoencoder = Model(input=[S, G], output=[S2, G2])
        encoder = Model(input=[S, G], output=emb_layer)

    return autoencoder, encoder

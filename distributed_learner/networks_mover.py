import tensorflow as tf
from keras.layers import Input, Dense, merge
from keras.models import Model


def create_actor_with_goal(state_size, goal_size, action_size):
    with tf.device("/cpu:0"):
        S = Input(shape=[state_size, ], name='actor_input')
        G = Input(shape=[goal_size, ], name='goal_input')
        S0 = Dense(50, init='random_normal', activation='relu')(S)
        G0 = Dense(10, init='random_normal', activation='relu')(G)
        h_start = merge([S0, G0], mode='concat')
        # HIDDEN1_UNITS=100, relu
        h0 = Dense(100, init='random_normal', activation='relu')(h_start)
        # HIDDEN2_UNITS=200, relu
        h1 = Dense(200, init='random_normal', activation='relu')(h0)
        # A = Activation('sigmoid')(BatchNormalization()(Dense(action_size, init='random_normal')(h1), training=True))
        # scale = tf.fill([None, action_size], 16)
        A = Dense(action_size, init='random_normal', activation='tanh')(h1)
        # output = merge([scale, A], mode='mul')
        model = Model(input=[S, G], output=A)
        action_gradient = tf.placeholder(tf.float32, [None, action_size])
        params_grad = tf.gradients(model.output, model.trainable_weights, -action_gradient)

    return model, action_gradient, params_grad


def create_critic_with_goal(state_size, goal_size, action_size):
    with tf.device("/cpu:0"):
        S = Input(shape=[state_size, ], name='actor_input')
        G = Input(shape=[goal_size, ], name='goal_input')
        S0 = Dense(50, init='random_normal', activation='relu')(S)
        G0 = Dense(10, init='random_normal', activation='relu')(G)
        h_start = merge([S0, G0], mode='concat')
        A = Input(shape=[action_size, ], name='action2')
        # HIDDEN1_UNITS, relu
        w1 = Dense(100, init='random_normal', activation='relu')(h_start)
        # HIDDEN2_UNITS, linear
        a1 = Dense(200, init='random_normal', activation='relu')(A)
        # HIDDEN2_UNITS, linear
        # h1 = Activation('tanh')(BatchNormalization()(Dense(200, init='random_normal')(w1), training=True))
        h1 = Dense(200, init='random_normal', activation='tanh')(w1)
        h2 = merge([h1, a1], mode='mul')
        # HIDDEN2_UNITS, linear
        h3 = Dense(200, init='random_normal', activation='relu')(h2)
        q = Dense(1, init='random_normal', activation='linear')(h3)
        model = Model(input=[S, G, A], output=q)
        action_grads = tf.gradients(model.output, A)

    return model, action_grads

import logging
import numpy as np
import threading
import time

import keras.backend as K
import tensorflow as tf
from scipy.io import savemat

from networks import create_actor, create_critic, create_emb_learner
from turn_env import AntTurnEnv

LEARNING_RATE = 0.001
NUM_CONCURRENT = 1

# global
NUM_EPOCHS = 30
NUM_EPISODES_PER_EPOCH = 3
MAX_EPISODE_LEN = 100
RESET_TARGET_NETWORK_EPOCHS = 1
MODEL_WEIGHTS_SUFFIX = "test0"
SUMMARY_PREFIX = "test0"

# main params
STATE_SIZE = 29
EMB_SIZE = 40
GOAL_SIZE = 1
ACTION_SIZE = 23

# env
VREP_PORT = 10000
PER_STEP_REWARD = -0.01
FINAL_REWARD = 10
TOLERANCE = 0.05
SPAWN_RADIUS = 6

g = tf.Graph()
sess = tf.Session()

thread_emb_loss = np.zeros((NUM_CONCURRENT, NUM_EPOCHS))
thread_critic_loss = np.zeros((NUM_CONCURRENT, NUM_EPOCHS))
thread_goals_achieved = np.zeros((NUM_CONCURRENT, NUM_EPOCHS))
thread_goals = np.zeros((NUM_CONCURRENT, NUM_EPOCHS, NUM_EPISODES_PER_EPOCH), dtype=np.float32)
thread_episodic_reward = np.zeros((NUM_CONCURRENT, NUM_EPOCHS, NUM_EPISODES_PER_EPOCH), dtype=np.float32)
thread_num_steps = np.zeros((NUM_CONCURRENT, NUM_EPOCHS, NUM_EPISODES_PER_EPOCH), dtype=np.int32)


def _learner_thread(thread_id, session_global, graph_ops):
    logger = logging.getLogger("learner")
    logger.info("STARTING THREAD:%d" % thread_id)

    # params tensors
    actor_params = graph_ops["actor_params"]
    critic_params = graph_ops["critic_params"]
    ae_params = graph_ops["ae_params"]
    e_params = graph_ops["e_params"]

    env = AntTurnEnv({
        'server_ip': '127.0.0.1',
        'server_port': VREP_PORT + thread_id,
        'vrep_exec_path': None,
        'vrep_scene_file': None,
        'per_step_reward': PER_STEP_REWARD,
        'final_reward': FINAL_REWARD,
        'tolerance': TOLERANCE,
        'spawn_radius': SPAWN_RADIUS
    })

    gamma = 0.99
    with tf.Graph().as_default(), tf.Session() as session:
        # make local networks
        lstate = tf.placeholder(tf.float32, [None, STATE_SIZE])
        lgoal = tf.placeholder(tf.float32, [None, GOAL_SIZE])
        laction = tf.placeholder(tf.float32, [None, ACTION_SIZE])
        lemb = tf.placeholder(tf.float32, [None, EMB_SIZE])

        lae_model, le_model = create_emb_learner(STATE_SIZE, GOAL_SIZE, EMB_SIZE)
        lactor, lact_grad, lactor_grads = create_actor(EMB_SIZE, ACTION_SIZE)
        lcritic, l_action_grads = create_critic(EMB_SIZE, ACTION_SIZE)

        # basics
        ltarget_state = tf.placeholder(tf.float32, [None, STATE_SIZE])
        ltarget_goal = tf.placeholder(tf.float32, [None, GOAL_SIZE])
        lae_values = lae_model([lstate, lgoal])
        lae_out_s, lae_out_g = lae_values[0], lae_values[1]
        lembs = le_model([lstate, lgoal])
        lq_values = lcritic([lemb, laction])
        lactor_action = lactor([lemb])

        # emb training
        emb_loss = tf.reduce_sum(tf.square(lae_out_s - ltarget_state)) + \
                   tf.reduce_sum(tf.square(lae_out_g - ltarget_goal))
        ae_grads = K.gradients(emb_loss, lae_model.trainable_weights)
        ae_full_grads = K.function(inputs=[lstate, lgoal, ltarget_state, ltarget_goal], outputs=ae_grads)

        # obtain critic grad_Q_wrt_a
        l_grad_Q_wrt_a = K.function(inputs=[lcritic.inputs[0], lcritic.inputs[1]], outputs=l_action_grads)
        # obtain full actor grads
        l_actor_full_grads = K.function(inputs=[lactor.inputs[0], lact_grad], outputs=lactor_grads)

        # obtain full critic gradients
        target_qs = tf.placeholder(tf.float32, [None, 1])
        critic_loss = tf.reduce_sum(tf.square(target_qs - lq_values))
        critic_grads = K.gradients(critic_loss, lcritic.trainable_weights)
        l_critic_full_grads = K.function(inputs=[lemb, laction, target_qs], outputs=critic_grads)

        lactor_params = lactor.trainable_weights
        lcritic_params = lcritic.trainable_weights
        lae_params = lae_model.trainable_weights
        lactor_set_params = lambda x: [lactor_params[i].assign(x[i]) for i in range(len(x))]
        lcritic_set_params = lambda x: [lcritic_params[i].assign(x[i]) for i in range(len(x))]
        lae_set_params = lambda x: [lae_params[i].assign(x[i]) for i in range(len(x))]

        session.run(tf.global_variables_initializer())

        logger.info("Obtaining target network params")
        local_params = [session_global.run(param) for param in ae_params]
        lae_set_params(local_params)
        local_params = [session_global.run(param) for param in actor_params]
        lactor_set_params(local_params)
        local_params = [session_global.run(param) for param in critic_params]
        lcritic_set_params(local_params)
        num_goals_achieved = 0

        for epoch in range(0, NUM_EPOCHS):
            cur_states = np.zeros((0, 29))
            cur_goals = np.zeros((0, 1))
            next_states = np.zeros((0, 29))
            next_goals = np.zeros((0, 1))
            actions = np.zeros((0, 23))
            rewards = np.zeros((0, 1))
            non_terminal = np.zeros((0, 1))

            for episode in range(0, NUM_EPISODES_PER_EPOCH):
                rand_goal = np.random.uniform(-np.pi, +np.pi)
                logger.info("START:epoch:%d, episode:%d, goal:%f" % (epoch, episode, rand_goal))
                env.set_goal(rand_goal)
                # cur_goal = randomly generated starting goal
                cur_state, cur_goal = env.start()
                logger.info("cur_state:%s" % str(cur_state))
                thread_goals[thread_id, epoch, episode] = cur_goal
                next_state, next_goal = cur_state, cur_goal
                num_steps = 0
                total_reward = 0
                for step_no in range(0, MAX_EPISODE_LEN):
                    emb = session.run(lembs, feed_dict={lstate: cur_state, lgoal: cur_goal}).reshape((1, -1))
                    action = session.run(lactor_action, feed_dict={lemb: emb})[0].reshape((1, -1))
                    # NOTE : adding a small random noise to system
                    action += np.random.normal(loc=0.0, scale=0.2, size=action.shape)
                    next_state, next_goal, reward, has_ended = env.step(action)
                    logger.info("next_state:%s" % str(next_state))
                    cur_states = np.vstack((cur_states, cur_state))
                    next_states = np.vstack((next_states, next_state))
                    cur_goals = np.vstack((cur_goals, cur_goal))
                    next_goals = np.vstack((next_goals, next_goal))
                    rewards = np.vstack((rewards, reward))
                    if has_ended:
                        non_terminal = np.vstack((non_terminal, 0.0))
                        num_goals_achieved += 1
                    else:
                        non_terminal = np.vstack((non_terminal, 1.0))
                    actions = np.vstack((actions, action))

                    total_reward += reward
                    num_steps += 1
                    if has_ended:
                        logger.info("achieved goal")
                        break
                    cur_state = next_state
                    cur_goal = next_goal
                env.reset()
                thread_num_steps[thread_id, epoch, episode] = num_steps
                thread_episodic_reward[thread_id, epoch, episode] = total_reward
                logger.info("END:epoch:%d, episode:%d, goal:%f, num_steps:%d" % (epoch, episode, rand_goal, num_steps))
            logger.info("preparing updates")
            # embed cur and next values
            cur_embs = session.run(lembs, feed_dict={lstate: cur_states, lgoal: cur_goals})
            next_embs = session.run(lembs, feed_dict={lstate: next_states, lgoal: next_goals})
            # obtain critic q-values
            logger.info("critic targets")
            q_values = session.run(lq_values, feed_dict={lemb: next_embs, laction: actions}).reshape((-1, 1))
            # obtain targets for critic
            targets = rewards + gamma * np.multiply(non_terminal, q_values).reshape((-1, 1))

            # compute full emb grads
            logger.info("EMB update")
            full_emb_grads = ae_full_grads([cur_states, cur_goals, cur_states, cur_goals])
            graph_ops["ae_grad_copy"](full_emb_grads)
            session_global.run(graph_ops["ae_grad_apply"])

            # compute full actor and critic grads
            # actor update
            logger.info("ACTOR update")
            local_action_grads = l_grad_Q_wrt_a([cur_embs, actions])[0]
            full_local_actor_grads = l_actor_full_grads([cur_embs, local_action_grads])
            graph_ops["actor_grad_copy"](full_local_actor_grads)
            session_global.run(graph_ops["actor_grad_apply"])

            # critic update
            logger.info("CRITIC update")
            full_local_critic_grads = l_critic_full_grads([cur_embs, actions, targets])
            graph_ops["critic_grad_copy"](full_local_critic_grads)
            session_global.run(graph_ops["critic_grad_apply"])

            # compute losses for summary
            thread_critic_loss[thread_id, epoch] = session.run(critic_loss, feed_dict={lemb: cur_embs, laction: actions,
                                                                                       target_qs: targets})
            thread_emb_loss[thread_id, epoch] = session.run(emb_loss, feed_dict={lstate: cur_states, lgoal: cur_goals,
                                                                                 ltarget_state: cur_states,
                                                                                 ltarget_goal: cur_goals})
            thread_goals_achieved[thread_id, epoch] = 100 * num_goals_achieved * 1.0 / (
                (epoch + 1) * NUM_EPISODES_PER_EPOCH * 1.0)

            # if out of sync, re-sync target networks with server
            if np.mod(epoch, RESET_TARGET_NETWORK_EPOCHS) == 0:
                logger.info("RESET TARGET NETWORK PARAMS")
                local_params = [session_global.run(param) for param in ae_params]
                lae_set_params(local_params)
                local_params = [session_global.run(param) for param in actor_params]
                lactor_set_params(local_params)
                local_params = [session_global.run(param) for param in critic_params]
                lcritic_set_params(local_params)

    logger.info("EXITING THREAD:%d" % thread_id)


def setup_graph(state_size, goal_size, emb_size, action_size):
    # states = tf.placeholder(tf.float32, [None, STATE_SIZE])
    # goals = tf.placeholder(tf.float32, [None, GOAL_SIZE])
    # embs = tf.placeholder(tf.float32, [None, EMB_SIZE])
    # actions = tf.placeholder(tf.float32, [None, ACTION_SIZE])
    actor, action_grad_ph, actor_grads = create_actor(emb_size, action_size)
    critic, q_action_grads = create_critic(emb_size, action_size)
    ae_model, e_model = create_emb_learner(state_size, goal_size, emb_size)

    # params
    actor_params = actor.trainable_weights
    critic_params = critic.trainable_weights
    ae_params = ae_model.trainable_weights
    e_params = e_model.trainable_weights
    actor_grads = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in actor_params]
    actor_grad_copy = lambda x: [actor_grads[i].assign(x[i]) for i in xrange(len(x))]
    actor_grad_apply = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(zip(actor_grads, actor_params))
    critic_grads = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in critic_params]
    critic_grad_copy = lambda x: [critic_grads[i].assign(x[i]) for i in xrange(len(x))]
    critic_grad_apply = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(zip(critic_grads, critic_params))
    ae_grads = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in ae_params]
    ae_grad_copy = lambda x: [ae_grads[i].assign(x[i]) for i in xrange(len(x))]
    ae_grad_apply = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(zip(ae_grads, ae_params))

    graph_ops = {
        "actor_grads": actor_grads,  # grads for actor
        "actor_grad_copy": actor_grad_copy,
        "actor_grad_apply": actor_grad_apply,
        "critic_grads": critic_grads,  # grads for critic
        "critic_grad_copy": critic_grad_copy,
        "critic_grad_apply": critic_grad_apply,
        "ae_grads": ae_grads,  # grads for emb
        "ae_grad_copy": ae_grad_copy,
        "ae_grad_apply": ae_grad_apply,
        "actor_params": actor_params,  # network params
        "critic_params": critic_params,
        "ae_params": ae_params,
        "e_params": e_params
    }
    return (actor, critic, ae_model), graph_ops


def train(session, models, graph_ops):
    session.run(tf.global_variables_initializer())
    # writer = tf.summary.FileWriter(SUMMARY_SAVE_PATH, session.graph)
    actor_learner_threads = [threading.Thread(target=_learner_thread, args=(thread_id, session, graph_ops)) for
                             thread_id in range(NUM_CONCURRENT)]
    for t in actor_learner_threads:
        t.start()

    done = False
    while not done:
        done = True
        for t in actor_learner_threads:
            if t.isAlive():
                done = False
                break

        # save global params
        models[0].save_weights('actor_server_' + MODEL_WEIGHTS_SUFFIX + '.h5')
        models[1].save_weights('critic_server_' + MODEL_WEIGHTS_SUFFIX + '.h5')
        models[2].save_weights('autoencoder_server_' + MODEL_WEIGHTS_SUFFIX + '.h5')
        savemat(SUMMARY_PREFIX + "_summary.mat",
                mdict={'thread_goals_achieved': thread_goals_achieved, 'thread_critic_loss': thread_critic_loss,
                       'thread_emb_loss': thread_emb_loss, 'thread_episodic_reward': thread_episodic_reward,
                       'thread_goals': thread_goals, 'thread_num_steps': thread_num_steps})

        # wake up every 60 seconds
        time.sleep(60)

    # save global params one last time
    models[0].save_weights('actor_server_' + MODEL_WEIGHTS_SUFFIX + '.h5')
    models[1].save_weights('critic_server_' + MODEL_WEIGHTS_SUFFIX + '.h5')
    models[2].save_weights('autoencoder_server_' + MODEL_WEIGHTS_SUFFIX + '.h5')

    # save summaries
    savemat(SUMMARY_PREFIX + "_summary.mat",
            mdict={'thread_goals_achieved': thread_goals_achieved, 'thread_critic_loss': thread_critic_loss,
                   'thread_emb_loss': thread_emb_loss, 'thread_episodic_reward': thread_episodic_reward,
                   'thread_goals': thread_goals, 'thread_num_steps': thread_num_steps})


def main(_):
    logger = logging.getLogger("learner")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler("log_file.txt")
    formatter = logging.Formatter("%(levelname)s:%(thread)d:%(filename)s:%(funcName)s:%(asctime)s::%(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # g = tf.Graph()
    with g.as_default(), sess as session:
        K.set_session(session)
        models, graph_ops = setup_graph(STATE_SIZE, GOAL_SIZE, EMB_SIZE, ACTION_SIZE)
        # attempt to load params here
        try:
            models[0].load_weights('actor_server_' + MODEL_WEIGHTS_SUFFIX + '.h5')
            models[1].load_weights('critic_server_' + MODEL_WEIGHTS_SUFFIX + '.h5')
            models[2].load_weights('autoencoder_server_' + MODEL_WEIGHTS_SUFFIX + '.h5')
        except:
            logger.error("failed to load all params")
        train(session, models, graph_ops)


if __name__ == "__main__":
    tf.app.run(main=main)

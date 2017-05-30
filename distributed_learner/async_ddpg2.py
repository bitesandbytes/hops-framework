import argparse
import logging
import numpy as np
import os
import threading
import time

import keras.backend as K
import tensorflow as tf

from networks import create_actor, create_critic, create_emb_learner
from turn_env import AntTurnEnv

LEARNING_RATE = 0.001
NUM_CONCURRENT = 1

# global
NUM_EPOCHS = 2
NUM_EPISODES_PER_EPOCH = 2
MAX_EPISODE_LEN = 5
RESET_TARGET_NETWORK_EPOCHS = 2

# main params
STATE_SIZE = 29
EMB_SIZE = 30
GOAL_SIZE = 1
ACTION_SIZE = 23

# env
VREP_PORT = 10000
PER_STEP_REWARD = -0.01
FINAL_REWARD = 10
TOLERANCE = 0.087
SPAWN_RADIUS = 6


def _learner_thread(thread_id, session_global, graph_ops):
    logger = logging.getLogger("learner")
    logger.info("thread_id:%d" % thread_id)
    '''
    # feed tensors
    tstate = graph_ops["states"]
    taction = graph_ops["actions"]
    tq_values = graph_ops["q_values"]

    # critic ops
    critic_q_values = graph_ops["critic_q_values"]
    critic_q_a_grads = graph_ops["critic_q_a_grads"]
    critic_optimize = graph_ops["critic_optimize"]

    # actor ops
    actor_actions = graph_ops["actor_actions"]
    #actor_grads_op = graph_ops["actor_grads_op"]
    actor_optimize = graph_ops["actor_optimize"]
    '''
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
        lactor, lact_grad, lactor_grads = create_actor(EMB_SIZE, ACTION_SIZE, LEARNING_RATE)
        lcritic, l_action_grads = create_critic(EMB_SIZE, ACTION_SIZE, LEARNING_RATE)

        # basics
        ltarget_state = tf.placeholder(tf.float32, [None, STATE_SIZE])
        ltarget_goal = tf.placeholder(tf.float32, [None, GOAL_SIZE])
        lae_values = lae_model([lstate, lgoal])
        lae_out_s, lae_out_g = lae_values[0], lae_values[1]
        lembs = le_model([lstate, lgoal])
        lq_values = lcritic([lemb, laction])
        lactor_action = lactor([lemb])

        # emb training
        emb_loss = tf.reduce_mean(tf.square(lae_out_s - ltarget_state)) + \
                   tf.reduce_mean(tf.square(lae_out_g - ltarget_goal))
        ae_grads = K.gradients(emb_loss, lae_model.trainable_weights)
        ae_full_grads = K.function(inputs=[lstate, lgoal, ltarget_state, ltarget_goal], outputs=ae_grads)

        # obtain critic grad_Q_wrt_a
        l_grad_Q_wrt_a = K.function(inputs=[lcritic.inputs[0], lcritic.inputs[1]], outputs=l_action_grads)
        # obtain full actor grads
        l_actor_full_grads = K.function(inputs=[lactor.inputs[0], lact_grad], outputs=lactor_grads)

        # obtain full critic gradients
        target_qs = tf.placeholder(tf.float32, [None, 1])
        critic_loss = tf.reduce_mean(tf.square(target_qs - lq_values))
        critic_grads = K.gradients(critic_loss, lcritic.trainable_weights)
        l_critic_full_grads = K.function(inputs=[lemb, laction, target_qs], outputs=critic_grads)

        lactor_params = lactor.trainable_weights
        lcritic_params = lcritic.trainable_weights
        lae_params = lae_model.trainable_weights
        lactor_set_params = lambda x: [lactor_params[i].assign(x[i]) for i in range(len(x))]
        lcritic_set_params = lambda x: [lcritic_params[i].assign(x[i]) for i in range(len(x))]
        lae_set_params = lambda x: [lae_params[i].assign(x[i]) for i in range(len(x))]

        session.run(tf.global_variables_initializer())

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
                next_state, next_goal = cur_state, cur_goal
                num_steps = 0
                for step_no in range(0, MAX_EPISODE_LEN):
                    emb = session.run(lembs, feed_dict={lstate: cur_state, lgoal: cur_goal}).reshape((1, -1))
                    action = session.run(lactor_action, feed_dict={lemb: emb})[0].reshape((1, -1))
                    # NOTE : adding a small random noise to system
                    action += np.random.normal(loc=0.0, scale=0.2, size=action.shape)
                    next_state, next_goal, reward, has_ended = env.step(action)
                    cur_states = np.vstack((cur_states, cur_state))
                    next_states = np.vstack((next_states, next_state))
                    cur_goals = np.vstack((cur_goals, cur_goal))
                    next_goals = np.vstack((next_goals, next_goal))
                    rewards = np.vstack((rewards, reward))
                    if has_ended:
                        non_terminal = np.vstack((non_terminal, 0.0))
                    else:
                        non_terminal = np.vstack((non_terminal, 1.0))
                    actions = np.vstack((actions, action))
                    num_steps += 1
                    if has_ended:
                        logger.info("achieved goal")
                    break
                    cur_state = next_state
                    cur_goal = next_goal
                env.reset()
                logger.info("END:epoch:%d, episode:%d, goal:%f, num_steps:%d" % (epoch, episode, rand_goal, num_steps))

            # embed cur and next values
            cur_embs = session.run(lembs, feed_dict={lstate: cur_states, lgoal: cur_goals})
            next_embs = session.run(lembs, feed_dict={lstate: next_states, lgoal: next_goals})
            # obtain critic q-values
            q_values = session.run(lq_values, feed_dict={lemb: next_embs, laction: actions}).reshape((-1, 1))
            # obtain targets for critic
            targets = rewards + gamma * np.multiply(non_terminal, q_values).reshape((-1, 1))

            # compute full emb grads
            full_emb_grads = ae_full_grads([cur_states, cur_goals, cur_states, cur_goals])
            graph_ops["ae_grad_copy"](full_emb_grads)
            session_global.run(graph_ops["ae_grad_apply"])

            # compute full actor and critic grads
            # actor update
            local_action_grads = l_grad_Q_wrt_a([cur_embs, actions])[0]
            full_local_actor_grads = l_actor_full_grads([cur_embs, local_action_grads])
            graph_ops["actor_grad_copy"](full_local_actor_grads)
            session_global.run(graph_ops["actor_grad_apply"])

            # critic update
            full_local_critic_grads = l_critic_full_grads([cur_embs, actions, targets])
            graph_ops["critic_grad_copy"](full_local_critic_grads)
            session_global.run(graph_ops["critic_grad_apply"])

            # if out of sync, re-sync target networks with server
            if np.mod(epoch, RESET_TARGET_NETWORK_EPOCHS) == 0:
                local_params = [session_global.run(param) for param in ae_params]
                lae_set_params(local_params)
                local_params = [session_global.run(param) for param in actor_params]
                lactor_set_params(local_params)
                local_params = [session_global.run(param) for param in critic_params]
                lcritic_set_params(local_params)

            # logger.info("bulk_q_values.shape:%s" % str(bulk_q_values.shape))

            # get action grads
            # epoch_critic_a_grads = session.run(critic_a_grads, feed_dict={state:cur_states, action:actions})[0]
            # epoch_actor_grads = session.run(, feed_dict={})
            '''
            # EMB train & update
            # train EMB
            embedder.fit(states=cur_states, goals=cur_goals)
            logger.info("EMB fit")
            # EMB update
            emb_lock.acquire()
            
            cur_weights = embedder.autoencoder.get_weights()
            target_weights = emb_server_net.autoencoder.get_weights()
            for i in xrange(len(cur_weights)):
                target_weights[i] = (1 - update_rate) * target_weights[i] + update_rate * cur_weights[i]
    
            emb_server_net.autoencoder.set_weights(target_weights)
            emb_local = embedder.autoencoder.get_weights()
            for i in xrange(len(emb_server)):
                emb_server[i] = ((1 - update_rate) * emb_server[i] + update_rate * emb_local[i]).reshape(
                    emb_server[i].shape)
            emb_out_of_sync += 1
    
            if emb_out_of_sync == emb_max_out_of_sync:
                embedder.autoencoder.set_weights(emb_server)
                emb_out_of_sync = 0
                logger.info("EMB synced")
            emb_lock.release()
            # DDPG train & update
            # DDPG train
            cur_embs, next_embs = embedder.embed(cur_states, cur_goals), embedder.embed(next_states, next_goals)
            # NOTE : multiplying by has_ended makes sure that final transition only takes reward as target
            targets = rewards + gamma * np.multiply(is_final, critic.target_model.predict(
                [next_embs, actor.target_model.predict(next_embs)]))
            loss = critic.model.train_on_batch([cur_embs, actions], targets)
            logger.info("critic loss:%f" % loss)
            actions_for_gradients = actor.model.predict(cur_embs)
            grad_q_wrt_a = critic.gradients(cur_embs, actions_for_gradients)
            actor.train(cur_embs, grad_q_wrt_a)
            actor.target_train()
            critic.target_train()
            logger.info("DDPG fit")
            # DDPG update
            ddpg_lock.acquire()
            # update server
            actor_local = actor.model.get_weights()
            for i in xrange(len(actor_server)):
                actor_server[i] = ((1 - update_rate) * actor_server[i] + update_rate * actor_local[i]).reshape(
                    actor_server[i].shape)
            critic_local = critic.model.get_weights()
            for i in xrange(len(critic_server)):
                critic_server[i] = ((1 - update_rate) * critic_server[i] + update_rate * critic_local[i]).reshape(
                    critic_server[i].shape)
    
            ddpg_out_of_sync += 1
            if ddpg_out_of_sync == ddpg_max_out_of_sync:
                actor.target_model.set_weights(actor_server)
                actor.model.set_weights(actor_server)
                critic.target_model.set_weights(critic_server)
                critic.model.set_weights(critic_server)
                ddpg_out_of_sync = 0
                logger.info("DDPG synced")
            ddpg_lock.release()
            
            # save network params
            if np.mod(epoch, args['save_every_x_epochs']) == 0:
                embedder.autoencoder.save_weights(
                    'embedder_thread' + str(thread_id) + args['model_weights_suffix'] + '.h5')
                actor.model.save_weights('actor_thread' + str(thread_id) + args['model_weights_suffix'] + '.h5')
                actor.target_model.save_weights('actor_thread' + str(thread_id) + args['model_weights_suffix'] + '.h5')
                critic.model.save_weights('critic_thread' + str(thread_id) + args['model_weights_suffix'] + '.h5')
                critic.target_model.save_weights(
                    'critic_thread' + str(thread_id) + args['model_weights_suffix'] + '.h5')
                logger.info("params saved")
    
            if np.mod(epoch, args['eval_every_x_epochs']) == 0:
                logger.info("model eval at epoch %d" % epoch)
                # average of 5 tries
                avg_reward = 0
                avg_num_steps = 0
                for t in range(0, 5):
                    env.reset()
                    rand_goal = np.random.uniform(-np.pi, +np.pi)
                    env.set_goal(rand_goal)
                    cur_state, cur_goal = env.start()
                    next_state, next_goal = cur_state, cur_goal
                    episode_reward = 0
                    num_steps = 0
                    for step in range(0, args['max_episode_length']):
                        action = actor.model.predict(embedder.embed(cur_state, cur_goal))
                        next_state, next_goal, reward, has_ended = env.step(action)
                        episode_reward += reward
                        num_steps += 1
                        if has_ended:
                            break
                    avg_reward += episode_reward
                    avg_num_steps += num_steps
                avg_reward /= 5.0
                avg_num_steps /= 5.0
                logger.info("avg_reward:%f, avg_num_steps:%f" % (avg_reward, avg_num_steps))
    
        # Save network params one last time
        embedder.autoencoder.save_weights('embedder_thread' + str(thread_id) + args['model_weights_suffix'] + '.h5')
        actor.model.save_weights('actor_thread' + str(thread_id) + args['model_weights_suffix'] + '.h5')
        actor.target_model.save_weights('actor_thread' + str(thread_id) + args['model_weights_suffix'] + '.h5')
        critic.model.save_weights('critic_thread' + str(thread_id) + args['model_weights_suffix'] + '.h5')
        critic.target_model.save_weights('critic_thread' + str(thread_id) + args['model_weights_suffix'] + '.h5')
        logger.info("learning complete; thread exiting")
        '''
        logger.info("EXIT")


def setup_graph(state_size, goal_size, emb_size, action_size):
    # states = tf.placeholder(tf.float32, [None, STATE_SIZE])
    # goals = tf.placeholder(tf.float32, [None, GOAL_SIZE])
    # embs = tf.placeholder(tf.float32, [None, EMB_SIZE])
    # actions = tf.placeholder(tf.float32, [None, ACTION_SIZE])
    actor, action_grad_ph, actor_grads = create_actor(emb_size, action_size, LEARNING_RATE)
    critic, q_action_grads = create_critic(emb_size, action_size, LEARNING_RATE)
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
    return graph_ops


'''
# placeholders
"states": states,
"actions": actions,
"q_values": q_values,
# actor ops
"actor_actions": actor_actions,
#"actor_grads_op": actor_grads_op,
"actor_optimize": actor_optimize,
# critic ops
"critic_q_values": q_values,
"critic_q_a_grads": critic_q_a_grads,
"critic_optimize": critic.train_on_batch,
'''


def train(session, graph_ops):
    session.run(tf.global_variables_initializer())
    # writer = tf.summary.FileWriter(SUMMARY_SAVE_PATH, session.graph)
    actor_learner_threads = [threading.Thread(target=_learner_thread, args=(thread_id, session, graph_ops)) for
                             thread_id in range(NUM_CONCURRENT)]
    for t in actor_learner_threads:
        t.start()

    for t in actor_learner_threads:
        t.join()


def other_func():
    os.chdir("../")
    # init parser
    parser = argparse.ArgumentParser(description="Parallel Asynchronous DDPG with Embedding Learner")
    parser.add_argument("--model_weights_suffix",
                        help="stand-in for <SUFFIX> in \"actor_<SUFFIX>\" and \"critic_<SUFFIX>\"")
    parser.add_argument("--actor_learning_rate", type=float, help="learning rate for actor network")
    parser.add_argument("--critic_learning_rate", type=float, help="learning rate for critic network")
    parser.add_argument("--emb_learning_rate", type=float, help="learning rate for embedding learner network")
    parser.add_argument("--emb_size", type=int, help="embedding dimension")
    parser.add_argument("--target_network_update_rate", type=float,
                        help="update rate to update target networks using current networks")
    parser.add_argument("--ddpg_max_out_of_sync", type=int,
                        help="DDPG - no. of episodes after which updates are pushed to param server, simultaneously "
                             "reading from it")
    parser.add_argument("--emb_max_out_of_sync", type=int,
                        help="EMB - no. of episodes after which updates are pushed to param server, simultaneously "
                             "reading from it")
    parser.add_argument("--num_learners", type=int,
                        help="no. of learning threads to use; each one also uses its own env")
    parser.add_argument("--batch_size", type=int, help="batch size for both DDPG and EMB")
    parser.add_argument("--num_epochs", type=int, help="number of training epochs")
    parser.add_argument("--num_episodes_per_epoch", type=int, help="number of episodes to run per epoch of training")
    parser.add_argument("--max_episode_length", type=int, help="max length of episode before terminatiion")
    parser.add_argument("--vrep_port_begin", type=int, help="remote port number to start V-REP simulator on")
    parser.add_argument("--vrep_scene_file", help="scene file(.ttt) for V-REP")
    parser.add_argument("--server_update_rate", type=float, help="update rate for server params")
    parser.add_argument("--save_every_x_epochs", type=int, help="number of epochs before saving params to file")
    parser.add_argument("--eval_every_x_epochs", type=int, help="number of epochs before eval model")
    parser.add_argument("--load_params", type=bool, help="loads params from file if True")
    parser.add_argument("--vrep_exec_path", help="full path for vrep.sh")
    parser.add_argument("--log_file", help="file to log to")

    args = parser.parse_args()

    # create TF session
    # sess = tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True))
    sess = tf.Session()

    # initialize logger
    logger = logging.getLogger("learner")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(args.log_file)
    formatter = logging.Formatter("%(levelname)s:%(thread)d:%(filename)s:%(funcName)s:%(asctime)s::%(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # other constants
    STATE_SIZE = 29
    GOAL_SIZE = 1
    ACTION_SIZE = 23

    # configs
    env_args = {
        'vrep_exec_path': args.vrep_exec_path,
        'vrep_port': args.vrep_port_begin,
        'vrep_scene_file': args.vrep_scene_file,
        'per_step_reward': -0.01,
        'final_reward': 10,
        'tolerance': 0.08,  # approx 5 degrees
        'spawn_radius': 6,
    }
    emb_args = {
        'sess': sess,
        'state_size': STATE_SIZE,
        'goal_size': GOAL_SIZE,
        'emb_size': args.emb_size,
        'batch_size': args.batch_size,
        'learning_rate': args.emb_learning_rate,
        # 'thread_idx' : 0,
    }
    actor_args = {
        'sess': sess,
        'state_size': args.emb_size,
        'action_size': ACTION_SIZE,
        'batch_size': args.batch_size,
        'target_update_rate': args.target_network_update_rate,
        'learning_rate': args.actor_learning_rate,
        'actor_network_config': {
            'hlayer_1_size': 100,
            'hlayer_1_type': 'relu',
            'hlayer_2_size': 200,
            'hlayer_2_type': 'relu'
        }
    }
    critic_args = {
        'sess': sess,
        'state_size': args.emb_size,
        'action_size': ACTION_SIZE,
        'batch_size': args.batch_size,
        'target_update_rate': args.target_network_update_rate,
        'learning_rate': args.critic_learning_rate,
        'critic_network_config': {
            'slayer1_size': 100,
            'slayer1_type': 'relu',
            'alayer_size': 200,
            'alayer_type': 'linear',
            'slayer2_size': 200,
            'slayer2_type': 'linear',
            'prefinal_layer_size': 200,
            'prefinal_layer_type': 'linear'
        }
    }
    thread_args = {
        'server_update_rate': args.server_update_rate,
        'emb_max_out_of_sync': args.emb_max_out_of_sync,
        'ddpg_max_out_of_sync': args.ddpg_max_out_of_sync,
        'actor_args': actor_args,
        'critic_args': critic_args,
        'emb_args': emb_args,
        'env_args': env_args,
        'gamma': 0.99,
        'model_weights_suffix': args.model_weights_suffix,
        'num_epochs': args.num_epochs,
        'state_size': STATE_SIZE,
        'goal_size': GOAL_SIZE,
        'action_size': ACTION_SIZE,
        'num_episodes_per_epoch': args.num_episodes_per_epoch,
        'max_episode_length': args.max_episode_length,
        'save_every_x_epochs': args.save_every_x_epochs,
        'load_params': args.load_params,
        'eval_every_x_epochs': args.eval_every_x_epochs
    }
    logger.info("args init complete")

    emb_lock, ddpg_lock = threading.Lock(), threading.Lock()
    server_emb, server_actor, server_critic = EmbeddingLearner(emb_args), ActorNetwork(actor_args), CriticNetwork(
        critic_args)
    # start learner threads
    emb_params, actor_params, critic_params = server_emb.autoencoder.get_weights(), server_actor.model.get_weights(), server_critic.model.get_weights()

    if args.load_params:
        try:
            emb_params = np.load('embedder_server' + args.model_weights_suffix + '.npy')
            actor_params = np.load('actor_server' + args.model_weights_suffix + '.npy')
            critic_params = np.load('critic_server' + args.model_weights_suffix + '.npy')
            # server_emb.autoencoder.load_weights('embedder_server' + args.model_weights_suffix + '.h5')
            # server_actor.model.load_weights('actor_server' + args.model_weights_suffix + '.h5')
            # server_critic.model.load_weights('critic_server' + args.model_weights_suffix + '.h5')
        except:
            logger.error("failed to load server params; re-initializing server params")

    threads = []
    for thread_id in range(0, args.num_learners):
        thread = threading.Thread(target=_learner_thread, args=(
            thread_args, sess, thread_id, (emb_params, actor_params, critic_params), (emb_lock, ddpg_lock)))
        threads.append(thread)
        thread.start()
    logger.info("started %d learner threads" % (len(threads)))

    # sleep for a minute, save params every minute
    done = False
    while not done:
        # just sleep for a while
        time.sleep(100)

        # check if all child threads are done
        done = True
        for thread in threads:
            if thread.isAlive():
                done = False
                break

        # save params
        emb_lock.acquire()
        # server_emb.autoencoder.set_weights(emb_params)
        np.save('embedder_server' + args.model_weights_suffix + '.npy', emb_params)
        # server_emb.autoencoder.save_weights('embedder_server' + args.model_weights_suffix + '.h5')
        emb_lock.release()
        ddpg_lock.acquire()
        # server_actor.model.set_weights(actor_params)
        np.save('actor_server' + args.model_weights_suffix + '.npy', actor_params)
        # server_actor.model.save_weights('actor_server' + args.model_weights_suffix + '.h5')
        # server_critic.model.set_weights(critic_params)
        np.save('critic_server' + args.model_weights_suffix + '.npy', critic_params)
        # server_critic.model.save_weights('critic_server' + args.model_weights_suffix + '.h5')
        ddpg_lock.release()
        logger.info("saving server params")


def main(_):
    logger = logging.getLogger("learner")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler("log_file.txt")
    formatter = logging.Formatter("%(levelname)s:%(thread)d:%(filename)s:%(funcName)s:%(asctime)s::%(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    g = tf.Graph()
    with g.as_default(), tf.Session() as session:
        K.set_session(session)
        graph_ops = setup_graph(STATE_SIZE, GOAL_SIZE, EMB_SIZE, ACTION_SIZE)
        train(session, graph_ops)


if __name__ == "__main__":
    tf.app.run(main=main)

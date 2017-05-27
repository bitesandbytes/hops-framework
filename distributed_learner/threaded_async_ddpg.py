import argparse
import logging
import threading
import time

import numpy as np
import tensorflow as tf

from actor_keras import ActorNetwork
from critic_keras import CriticNetwork
from embedding_learner import EmbeddingLearner
from scenes.turn_env import AntTurnEnv


def _learner_thread(args, thread_id, sync_nets, locks):
    logger = logging.getLogger("learner")

    emb_lock, ddpg_lock = locks[0], locks[1]
    emb_server_net, actor_server_net, critic_server_net = sync_nets[0], sync_nets[1], sync_nets[2]
    update_rate = args['server_update_rate']
    emb_out_of_sync, ddpg_out_of_sync = 0, 0
    emb_max_out_of_sync, ddpg_max_out_of_sync = args['emb_max_out_of_sync'], args['ddpg_max_out_of_sync']

    env_args = args['env_args']
    # init Actor, Critic and EMB networks
    actor = ActorNetwork(args['actor_args'])
    critic = CriticNetwork(args['critic_args'])
    embedder = EmbeddingLearner(args['emb_args'])
    env = AntTurnEnv({
        'server_ip': '127.0.0.1',
        'server_port': env_args['vrep_port'] + thread_id,
        'vrep_exec_path': env_args['vrep_exec_path'],
        'vrep_scene_file': env_args['vrep_scene_file'],
        'per_step_reward': env_args['per_step_reward'],
        'final_reward': env_args['final_reward'],
        'tolerance': env_args['tolerance'],
        'spawn_radius': env_args['spawn_radius']
    })

    gamma = args['gamma']

    # load N/W weights
    if args['load_params']:
        try:
            embedder.autoencoder.load_weights('embedder_thread' + str(thread_id) + args['model_weights_suffix'] + '.h5')
            actor.model.load_weights('actor_thread' + str(thread_id) + args['model_weights_suffix'] + '.h5')
            actor.target_model.load_weights('actor_thread' + str(thread_id) + args['model_weights_suffix'] + '.h5')
            critic.model.load_weights('critic_thread' + str(thread_id) + args['model_weights_suffix'] + '.h5')
            critic.target_model.load_weights('critic_thread' + str(thread_id) + args['model_weights_suffix'] + '.h5')
            logger.info("loaded weights from file, resuming training")
        except:
            logger.error("unable to load learner thread params; re-initializing")
            pass

    for epoch in range(0, args['num_epochs']):
        cur_states = np.zeros(0, args['state_size'])
        cur_goals = np.zeros(0, args['goal_size'])
        next_states = np.zeros(0, args['state_size'])
        next_goals = np.zeros(0, args['goal_size'])
        actions = np.zeros(0, args['action_size'])
        rewards = np.zeros(0, 1)
        is_final = np.zeros(0, 1)

        for episode in range(0, args['num_episodes_per_epoch']):
            env.reset()
            rand_goal = np.random.uniform(-np.pi, +np.pi)
            logger.info("starting - epoch:%d, episode:%d, goal:%f" % (epoch, episode, rand_goal[0]))
            env.set_goal(rand_goal)
            # cur_goal = randomly generated starting goal
            cur_state, cur_goal = env.start()
            next_state, next_goal = cur_state, cur_goal
            num_steps = 0
            for step_no in range(0, args['max_episode_length']):
                action = actor.target_model.predict(embedder.embed(cur_state, cur_goal))
                # NOTE : adding a small random noise to system
                action += np.random.normal(loc=0.0, scale=0.5, size=action.shape)
                next_state, next_goal, reward, has_ended = env.step(action)
                np.append(cur_states, cur_state, axis=0)
                np.append(next_states, next_state, axis=0)
                np.append(cur_goals, cur_goal, axis=0)
                np.append(next_goals, next_goal, axis=0)
                np.append(rewards, reward, axis=0)
                if has_ended:
                    np.append(is_final, np.asarray([0.0]), axis=0)
                else:
                    np.append(is_final, np.asarray([1.0]), axis=0)
                np.append(actions, action, axis=0)
                num_steps += 1
                if has_ended:
                    logger.info("achieved goal")
                    break
                cur_state = next_state
                cur_goal = next_goal
            logger.info(
                "ended - epoch:%d, episode:%d, goal:%f, num_steps:%d" % (epoch, episode, rand_goal[0], num_steps))
        # EMB train & update
        # train EMB
        embedder.fit(states=cur_states, goals=cur_goals)
        logger.info("EMB fit")
        # EMB update
        emb_lock.acquire()
        for cur_layer, server_layer in zip(embedder.autoencoder.layers, emb_server_net.autoencoder.layers):
            server_layer.set_weights(
                (1 - update_rate) * server_layer.get_weights() + update_rate * cur_layer.get_weights())
        emb_out_of_sync += 1
        if emb_out_of_sync == emb_max_out_of_sync:
            for cur_layer, server_layer in zip(embedder.autoencoder.layers, emb_server_net.layers):
                cur_layer.set_weights(server_layer.get_weights())
            emb_out_of_sync = 0
            logger.info("EMB synced")
        emb_lock.release()

        # DDPG train & update
        # DDPG train
        cur_embs, next_embs = embedder.embed(cur_states, cur_goals), embedder.embed(next_states, next_goals)
        # NOTE : multiplying by has_ended makes sure that final transition only takes reward as target
        targets = rewards + gamma * np.multiply(is_final, critic.target_model.predict(
            [next_embs, actor.target_model.predict(next_embs)]))
        critic.model.train_on_batch(cur_embs, targets)
        actions_for_gradients = actor.model.predict(cur_embs)
        grad_q_wrt_a = critic.gradients(cur_embs, actions_for_gradients)
        actor.train(cur_embs, grad_q_wrt_a)
        actor.target_train()
        critic.target_train()
        logger.info("DDPG fit")
        # DDPG update
        ddpg_lock.acquire()
        for cur_layer, server_layer in zip(actor.model.layers, actor_server_net.model.layers):
            server_layer.set_weights(
                (1 - update_rate) * server_layer.get_weights() + update_rate * cur_layer.get_weights())
        for cur_layer, server_layer in zip(critic.model.layers, critic_server_net.model.layers):
            server_layer.set_weights(
                (1 - update_rate) * server_layer.get_weights() + update_rate * cur_layer.get_weights())
        ddpg_out_of_sync += 1
        if ddpg_out_of_sync == ddpg_max_out_of_sync:
            for cur_layer, server_layer in zip(actor.model.layers, actor_server_net.model.layers):
                cur_layer.set_weights(server_layer.get_weights())
            for cur_layer, server_layer in zip(critic.model.layers, critic_server_net.model.layers):
                cur_layer.set_weights(server_layer.get_weights())
            ddpg_out_of_sync = 0
            logger.info("DDPG synced")
        ddpg_lock.release()

        # save network params
        if np.mod(epoch, args['save_every_x_epochs']) == 0:
            embedder.autoencoder.save_weights('embedder_thread' + str(thread_id) + args['model_weights_suffix'] + '.h5')
            actor.model.save_weights('actor_thread' + str(thread_id) + args['model_weights_suffix'] + '.h5')
            actor.target_model.save_weights('actor_thread' + str(thread_id) + args['model_weights_suffix'] + '.h5')
            critic.model.save_weights('critic_thread' + str(thread_id) + args['model_weights_suffix'] + '.h5')
            critic.target_model.save_weights('critic_thread' + str(thread_id) + args['model_weights_suffix'] + '.h5')
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


if __name__ == "__main__":
    # init parser
    parser = argparse.ArgumentParser(description="Parallel Asynchronous DDPG with Embedding Learner")
    parser.add_argument("model_weights_suffix",
                        help="stand-in for <SUFFIX> in \"actor_<SUFFIX>\" and \"critic_<SUFFIX>\"")
    parser.add_argument("actor_learning_rate", type=float, help="learning rate for actor network")
    parser.add_argument("critic_learning_rate", type=float, help="learning rate for critic network")
    parser.add_argument("emb_learning_rate", type=float, help="learning rate for embedding learner network")
    parser.add_argument("emb_size", type=int, help="embedding dimension")
    parser.add_argument("target_network_update_rate", type=float,
                        help="update rate to update target networks using current networks")
    parser.add_argument("ddpg_max_out_of_sync", type=int,
                        help="DDPG - no. of episodes after which updates are pushed to param server, simultaneously "
                             "reading from it")
    parser.add_argument("emb_max_out_of_sync", type=int,
                        help="EMB - no. of episodes after which updates are pushed to param server, simultaneously "
                             "reading from it")
    parser.add_argument("num_learners", type=int, help="no. of learning threads to use; each one also uses its own env")
    parser.add_argument("batch_size", type=int, help="batch size for both DDPG and EMB")
    parser.add_argument("num_epochs", type=int, help="number of training epochs")
    parser.add_argument("num_episodes_per_epoch", type=int, help="number of episodes to run per epoch of training")
    parser.add_argument("max_episode_length", help="max length of episode before terminatiion")
    parser.add_argument("vrep_port_begin", type=int, help="remote port number to start V-REP simulator on")
    parser.add_argument("vrep_scene_file", help="scene file(.ttt) for V-REP")
    parser.add_argument("server_update_rate", type=float, help="update rate for server params")
    parser.add_argument("save_every_x_epochs", type=int, help="number of epochs before saving params to file")
    parser.add_argument("eval_every_x_epochs", type=int, help="number of epochs before eval model")
    parser.add_argument("load_params", type=bool, help="loads params from file if True")
    parser.add_argument("vrep_exec_path", help="full path for vrep.sh")
    parser.add_argument("log_file", help="file to log to")

    args = parser.parse_args()

    # create TF session
    sess = tf.get_default_session()

    # initialize logger
    logger = logging.getLogger("learner")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(args.log_file)
    formatter = logging.Formatter("%(levelname)s:%(thread)d:%(filename)s:%(funcName)s:%(asctime)s::%(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # configs
    env_args = {
        'vrep_exec_path': args.vrep_exec_path,
        'vrep_port': args.vrep_port_begin,
        'vrep_scene_file': args.vrep_scene_file,
        'per_step_reward': -0.01,
        'final_reward': 1,
        'tolerance': 1,
        'spawn_radius': 6,
    }
    emb_args = {
        'sess': sess,
        'state_size': args.state_size,
        'goal_size': args.goal_size,
        'emb_size': args.emb_size,
        'batch_size': args.batch_size,
        'learning_rate': args.emb_learning_rate,
        # 'thread_idx' : 0,
    }
    actor_args = {
        'sess': sess,
        'state_size': args.emb_size,
        'action_size': args.action_size,
        'batch_size': args.batch_size,
        'target_update_rate': args.target_update_rate,
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
        'action_size': args.action_size,
        'batch_size': args.batch_size,
        'target_update_rate': args.target_update_rate,
        'learning_rate': args.critic_learning_rate,
        'critic_network_config': {
            'slayer_1_size': 100,
            'slayer_1_type': 'relu',
            'alayer_size': 100,
            'alayer_type': 'linear',
            'slayer_2_size': 200,
            'slayer_2_type': 'linear',
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
        'state_size': args.state_size,
        'goal_size': args.goal_size,
        'action_size': args.action_size,
        'num_episodes_per_epoch': args.num_episodes_per_epoch,
        'max_episode_length': args.max_episode_length,
        'save_every_x_epochs': args.thread_save_every_x_epochs,
        'load_params': args.load_params,
        'eval_every_x_epochs': args.eval_every_x_epochs
    }
    logger.info("args init complete")

    emb_lock, ddpg_lock = threading.Lock(), threading.Lock()
    server_emb, server_actor, server_critic = EmbeddingLearner(emb_args), ActorNetwork(actor_args), CriticNetwork(
        critic_args)
    if args.load_params:
        try:
            server_emb.autoencoder.load_weights('embedder_server' + args['model_weights_suffix'] + '.h5')
            server_actor.model.load_weights('actor_server' + args['model_weights_suffix'] + '.h5')
            server_critic.model.load_weights('critic_server' + args['model_weights_suffix'] + '.h5')
        except:
            logger.error("failed to load server params; re-initializing server params")

    # start learner threads
    threads = []
    for thread_id in range(0, args.num_learners):
        thread = threading.Thread(target=_learner_thread, args=(
            thread_args, thread_id, (server_emb, server_actor, server_critic), (emb_lock, ddpg_lock)))
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
        threads = [t for t in threads if not t.handled]

        # save params
        emb_lock.acquire()
        server_emb.autoencoder.save_weights('embedder_server' + args['model_weights_suffix'] + '.h5')
        emb_lock.release()
        ddpg_lock.acquire()
        server_actor.model.save_weights('actor_server' + args['model_weights_suffix'] + '.h5')
        server_critic.model.save_weights('critic_server' + args['model_weights_suffix'] + '.h5')
        ddpg_lock.release()

        logger.info("saving server params")

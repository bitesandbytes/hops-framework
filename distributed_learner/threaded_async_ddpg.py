import argparse
import threading
import numpy as np
import keras.backend as K

from actor_keras import ActorNetwork
from critic_keras import CriticNetwork
from embedding_learner import EmbeddingLearner

def _learner_thread(args, thread_id, sync_nets, locks):
    emb_lock, ddpg_lock = locks[0], locks[1]
    emb_server_net, actor_server_net, critic_server_net = sync_nets[0], sync_nets[1], sync_nets[2]
    update_rate = args['server_update_rate']
    emb_out_of_sync, ddpg_out_of_sync = 0, 0
    emb_max_out_of_sync, ddpg_max_out_of_sync = args['emb_max_out_of_sync'], args['ddpg_max_out_of_sync']

    # TODO : init env
    env_args = args['env_args']
    # init Actor, Critic and EMB networks
    actor = ActorNetwork(args['actor_args'])
    critic = CriticNetwork(args['critic_args'])
    embedder = EmbeddingLearner(args['emb_args'])
    gamma = args['gamma']
    # TODO : load network weights
    if args['load_params']:
        try:
            embedder.autoencoder.load_weights('embedder_'+args['model_weights_suffix']+'.h5')
            actor.model.load_weights('actor_'+args['model_weights_suffix']+'.h5')
            actor.target_model.load_weights('actor_'+args['model_weights_suffix']+'.h5')
            critic.model.load_weights('critic_'+args['model_weights_suffix']+'.h5')
            critic.target_model.load_weights('critic_'+args['model_weights_suffix']+'.h5')
        except:
            # TODO : log error here
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
            # TODO : reset env
            cur_state = env.start()
            next_state = cur_state
            # TODO : generate random goal
            # cur_goal = randomly generated starting goal
            cur_goal = None
            next_goal = cur_goal
            for step_no in range(0, args['max_episode_length']):
                action = actor.target_model.predict([cur_state, cur_goal]) # TODO : add noise to target model action
                next_state, reward, has_ended = env.step(action)
                # TODO : compute next goal
                np.append(cur_states, cur_state, axis=0)
                np.append(next_states, next_state, axis=0)
                np.append(cur_goals, cur_goal, axis=0)
                np.append(next_goals, next_goal, axis=0)
                np.append(rewards, reward, axis=0)
                np.append(1.0 if is_final else 0, has_ended, axis=0)
                np.append(actions, action, axis=0)
                if has_ended:
                    break
                cur_state = next_state
                cur_goal = next_goal

    # EMB train & update
    # train EMB
    embedder.fit(states=cur_states, goals=cur_goals)
    # EMB update
    emb_lock.acquire()
    for cur_layer, server_layer in zip(embedder.autoencoder.layers, emb_server_net.autoencoder.layers):
        server_layer.set_weights((1-update_rate)*server_layer.get_weights() + update_rate*cur_layer.get_weights())
    emb_out_of_sync += 1
    # TODO : log
    if emb_out_of_sync == emb_max_out_of_sync:
        for cur_layer, server_layer in zip(embedder.autoencoder.layers, emb_server_net.layers):
            cur_layer.set_weights(server_layer.get_weights())
        emb_out_of_sync = 0
    emb_lock.release()

    # DDPG train & update
    cur_embs, next_embs = embedder.embed(cur_states, cur_goals), embedder.embed(next_states, next_goals)
    targets = rewards + gamma*critic.target_model.predict([cur_embs, actor.target_model.predict(next_embs)])
    # DDPG update
    ddpg_lock.acquire()
    for cur_layer, server_layer in zip(actor.model.layers, actor_server_net.model.layers):
        server_layer.set_weights((1-update_rate)*server_layer.get_weights() + update_rate*cur_layer.get_weights())
    for cur_layer, server_layer in zip(critic.model.layers, critic_server_net.model.layers):
        server_layer.set_weights((1-update_rate)*server_layer.get_weights() + update_rate*cur_layer.get_weights())
    ddpg_out_of_sync += 1
    # TODO : log
    if ddpg_out_of_sync == ddpg_max_out_of_sync:
        for cur_layer, server_layer in zip(actor.model.layers, actor_server_net.model.layers):
            cur_layer.set_weights(server_layer.get_weights())
        for cur_layer, server_layer in zip(critic.model.layers, critic_server_net.model.layers):
            cur_layer.set_weights(server_layer.get_weights())
        ddpg_out_of_sync = 0
    ddpg_lock.release()


if __name__=="__main__":
    # init parser
    parser = argparse.ArgumentParser(description="Parallel Asynchronous DDPG with Embedding Learner")
    parser.add_argument("model_weights_suffix", help="stand-in for <SUFFIX> in \"actor_<SUFFIX>\" and \"critic_<SUFFIX>\"")
    parser.add_argument("actor_update_rate", type=float, help="update rate for actor network")
    parser.add_argument("critic_update_rate", type=float, help="update rate for critic network")
    parser.add_argument("target_network_update_rate", type=float, help="update rate to update target networks using current networks")
    parser.add_argument("ddpg_max_out_of_sync", type=int, help="DDPG - no. of episodes after which updates are pushed to param server, simultaneously reading from it")
    parser.add_argument("emb_max_out_of_sync", type=int, help="EMB - no. of episodes after which updates are pushed to param server, simultaneously reading from it")
    parser.add_argument("num_learners", type=int, help="no. of learning threads to use; each one also uses its own env")
    parser.add_argument("batch_size", type=int, help="batch size for both DDPG and EMB")
    parser.add_argument("vrep_port_begin", type=int, help="remote port number to start V-REP simulator on")
    parser.add_argument("vrep_scene_file", help="scene file(.ttt) for V-REP")
    parser.add_argument("max_episode_length", help="max length of episode before terminatiion")

    args = parser.parse_args()

    # TODO : generate thread_args, emb_args, actor_args, critic_args dicts
    # configs
    env_args = {
        vrep_port = args.vrep_port_begin,
        vrep_scene_file = args.vrep_scene_file,
        OTHER_ARGS_HERE = None,
    }
    emb_args = {
        'sess' : ,
        'state_size' : ,
        'goal_size' : ,
        'emb_size' : ,
        'batch_size' : ,
        'learning_rate' : ,
        'thread_idx' : ,
    }
    actor_args = {
        'sess' : ,
        'state_size' : ,  # same as emb_size
        'action_size' : ,
        'batch_size' : ,
        'target_update_rate' : ,
        'learning_rate' : ,
        'actor_network_config' : {
            'hlayer_1_size' : ,
            'hlayer_1_type' : 'relu',
            'hlayer_2_size' : ,
            'hlayer_2_type' : 'relu'
        }
    }
    critic_args = {
        'sess' : ,
        'state_size' : ,  # same as emb_size
        'action_size' : ,
        'batch_size' : ,
        'target_update_rate' : ,
        'learning_rate' : ,
        'critic_network_config' : {
            'slayer_1_size' : ,
            'slayer_1_type' : 'relu',
            'alayer_size' : ,
            'alayer_type' : 'linear',
            'slayer_2_size' : ,
            'slayer_2_type' : 'linear',
            'prefinal_layer_size' : ,
            'prefinal_layer_type' : 'linear'
        }
    }
    thread_args = {
        'server_update_rate' : ,
        'emb_max_out_of_sync' : ,
        'ddpg_max_out_of_sync' : ,
        'actor_args' : ,
        'critic_args' : ,
        'emb_args' : ,
        'gamma' : ,
        'model_weights_suffix' : ,
        'num_epochs' = ,
        'state_size' = ,
        'goal_size' = ,
        'action_size' = ,
        'num_episodes_per_epoch' = ,
        'max_episode_length' = ,

    }
    emb_lock, ddpg_lock = threading.Lock(), threading.Lock()
    server_emb, server_actor, server_critic = EmebddingLearner(emb_args), ActorNetwork(actor_args), CriticNetwork(critic_args)

    # start learner threads
    for thread_id in range(0, args.num_learners):
        thread = threading.Thread(target=_learner_thread, args=(args, thread_id, (server_emb, server_actor, server_critic), (emb_lock, ddpg_lock)))
        thread.start()

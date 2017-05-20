import tensorflow as tf
import numpy as np
import random
import argparse

from replay_buffer import ReplayBuffer
from actor_keras import ActorNetwork
from critic_keras import CriticNetwork

def learn(args):
    # init constants from args
    # init keras/TF
    onfig = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)

    # init actor-critic, replay buffer
    actor=ActorNetwork(sess, state_dim, action_dim, )
    # init env
    # init param client
    # load checkpoint
    while episode_count < MAX_EPISODES:
        # run an episode using target networks
        # add experience to replay buffer
        # sample experience
        # train embedding
        # obtain embeddings
        # train actor-critic networks
        # reset env to start state
        # send updates to param server
        # sync with param server every 5 episodes
        pass
    pass

if __name__=="__main__":
    # init parser
    parser = argparse.ArgumentParser(description="DDPG with Embedding Learner")
    parser.add_argument("model_weights_suffix", help="stand-in for <SUFFIX> in \"actor_<SUFFIX>\" and \"critic_<SUFFIX>\"")
    parser.add_argument("actor_update_rate", type=float, help="update rate for actor network")
    parser.add_argument("critic_update_rate", type=float, help="update rate for critic network")
    parser.add_argument("num_episodes_per_update", type=int, help="number of episodes to run before making an update")
    parser.add_argument("ddpg_max_out_of_sync", type=int, help="DDPG - max number of updates to parameter server before asking for sync")
    parser.add_argument("emb_max_out_of_sync", type=int, help="EMB - max number of updates to parameter server before asking for sync")
    parser.add_argument("vrep_port", type=int, help="port on which vrep is running on localhost")
    parser.add_argument("ddpg_server_ip", help="IP of parameter server for DDPG")
    parser.add_argument("emb_server_ip", help="IP of parameter server for EMB")
    parser.add_argument("ddpg_server_update_port", type=int, help="port # for updates to DDPG parameter server")
    parser.add_argument("ddpg_server_sync_port", type=int, help="port # for sync with DDPG parameter server")
    parser.add_argument("emb_server_update_port", type=int, help="port # for updates to EMB parameter server")
    parser.add_argument("emb_server_sync_port", type=int, help="port # for sync with EMB parameter server")
    # start learning
    pass

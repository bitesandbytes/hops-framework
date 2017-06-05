import numpy as np

from scipy.io import savemat

from networks import *
from turn_env import AntTurnEnv

# main params
STATE_SIZE = 29
EMB_SIZE = 40
GOAL_SIZE = 1
ACTION_SIZE = 23

# eval paras
NUM_EPISODES = 10
MAX_EPISODE_LEN = 1000

if __name__ == "__main__":
    env = AntTurnEnv({
        'server_ip': '127.0.0.1',
        'server_port': 10000,
        'vrep_exec_path': None,
        'vrep_scene_file': None,
        'per_step_reward': -0.01,
        'final_reward': 10.0,
        'tolerance': 0.05,
        'spawn_radius': 6.0
    })

    a = create_actor(EMB_SIZE, ACTION_SIZE)
    e = create_emb_learner(STATE_SIZE, GOAL_SIZE, EMB_SIZE)
    c = create_critic(EMB_SIZE, ACTION_SIZE)
    actor = a[0]
    encoder = e[1]
    autoencoder = e[0]
    critic = c[0]
    try:
        actor.load_weights("actor_server_test0.h5")
        critic.load_weights("critic_server_test0.h5")
        autoencoder.load_weights("autoencoder_server_test0.h5")
    except:
        print("unable to load weights")

    episodic_reward = np.zeros((1, NUM_EPISODES))
    episode_num_steps = np.zeros((1, NUM_EPISODES))
    episode_goals = np.zeros((1, NUM_EPISODES))
    for i in range(0, NUM_EPISODES):
        print("Starting episode %d" % (i + 1))
        rand_goal = np.random.uniform(-np.pi, +np.pi)
        env.set_goal(rand_goal)
        cur_state, cur_goal = env.start()
        episode_goals[i] = cur_goal
        for j in range(0, MAX_EPISODE_LEN):
            emb = encoder.predict([cur_state, cur_goal])
            action = actor.predict([emb])
            next_state, next_goal, reward, is_done = env.step(action)
            episodic_reward[i] += reward
            episode_num_steps[i] += 1
            if is_done:
                break
            cur_state = next_state
            cur_goal = next_goal
        env.reset()

    # save results to matlab file
    savemat("eval_turn_env.mat", mdict={"episodic_reward": episodic_reward, "episode_num_steps": episode_num_steps,
                                        "episode_goals": episode_goals})

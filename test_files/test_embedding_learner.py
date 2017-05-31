from collections import deque

from actor.embedding_learner import EmbeddingLearner
from scenes.asti import Asti

from actor.random_policy import RandomDeterministicPolicy

if __name__ == '__main__':
    # 26 dims = 20 state dims, 3 euler angles, 3 cartesian coords
    emb_leaner = EmbeddingLearner(inp_dim=26, emb_dim=10, learning_rate=0.001)
    policy = RandomDeterministicPolicy(26, 20)
    env = Asti(server_ip='127.0.0.1', server_port=19997)
    env.init_client()

    # Collect 1M transitions, 1000 transitions max per trajectory
    cur_time = 0.0
    max_trans = 1 ** 6
    custom_dt = 0.005

    # Set 5ms as custom dt for simulations
    env.set_custom_dt(custom_dt)
    transitions = deque()
    prev_state = np.append(np.append(env.get_joint_pos(), env.get_position(), axis=0), env.get_orientation(), axis=0)
    for i in range(0, max_trans):
        forces = policy.step(prev_state)
        env.set_forces_and_trigger(forces)
        new_state = np.append(np.append(env.get_joint_pos(), env.get_position(), axis=0), env.get_orientation(), axis=0)
        transitions.append((prev_state, forces, 0, new_state))
        prev_state = new_state
        cur_time += custom_dt
        if i / 100 != 0:
            print "Step #%s" % (i)

    env.stop_simulation()

import logging
import numpy as np
import time

from hexapod import Ant

if __name__ == "__main__":
    logger = logging.getLogger("learner")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler("logfile.txt")
    formatter = logging.Formatter("%(levelname)s:%(thread)d:%(filename)s:%(funcName)s:%(asctime)s::%(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ant = Ant("127.0.0.1", 10000)
    ant.init_client()
    ant.stop_simulation()

    # perform one thousand steps
    for i in range(0, 10):
        time.sleep(1)
        rand_pos = np.random.uniform(-6, +6, (1, 2))
        ant.start_simulation()
        ant.set_position_and_rotation(np.random.uniform(-6, +6, (2,)), np.random.uniform(-np.pi, +np.pi))
        for i in range(0, 100):
            logger.info("step:%d" % i)
            state = np.hstack((ant.get_joint_pos(), ant.get_orientation(), ant.get_position()))
            force_vec = np.random.uniform(-1, +1, (1, ant.joint_count))
            ant.set_forces_and_trigger(force_vec)
            # time.sleep(0.050)
        ant.stop_simulation()

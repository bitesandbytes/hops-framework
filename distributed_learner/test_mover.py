import logging
import numpy as np
import time

from mover import Mover

if __name__ == "__main__":
    logger = logging.getLogger("learner")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler("logfile.txt")
    formatter = logging.Formatter("%(levelname)s:%(thread)d:%(filename)s:%(funcName)s:%(asctime)s::%(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    mov = Mover("127.0.0.1", 10000)
    mov.init_client()
    print('ori' + str(mov.get_orientation()))
    print('pos' + str(mov.get_position()))
    mov.stop_simulation()
    # mov.stop_simulation()

    # perform one thousand steps
    for i in range(0, 10):
        rand_pos = np.random.uniform(-0.2, +0.2, (2,))
        time.sleep(0.5)
        mov.start_simulation()
        mov.set_position_and_rotation(rand_pos, np.random.uniform(-np.pi, +np.pi))
        # mov.set_target_vel(np.zeros(2))
        print('ori' + str(mov.get_orientation()))
        print('pos' + str(mov.get_position()))
        time.sleep(1)
        for k in range(0, 100):
            logger.info("step:%d" % i)
            state = np.hstack((mov.get_joint_pos(), mov.get_joint_vel()))
            force_vec = np.random.uniform(-10, +10, (mov.joint_count,))
            # print(force_vec.shape)
            mov.set_target_vel(force_vec)
            # time.sleep(0.050)
        mov.stop_simulation()

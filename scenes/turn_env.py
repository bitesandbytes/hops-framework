import logging
import numpy as np
from hexapod import Ant
from env import BaseEnv
import subprocess

class AntTurnEnv(BaseEnv):
    def __init__(self, args):
        # start a VREP headless background process
        subprocess.Popen([args['vrep_exec_path'], '-h', '-gREMOTEAPISERVERSERVICE_'+str(args['server_port'])+'_FALSE_TRUE', args['vrep_scene_file']])
        self.ant = Ant(args['server_ip'], args['server_port'])
        self.ant.init_client()
        self.per_step_reward = args['per_step_reward']
        self.final_reward = args['final_reward']
        self.tolerance = args['tolerance']
        self.server_ip, self.server_port = args['server_ip'], args['server_port']
        self.spawn_radius = args['spawn_radius']

    def set_goal(self, delta_theta):
        self.delta_theta = delta_theta

    def start(self):
        logging.getLogger("learner").info("starting env with goal:%f" % (self.delta_theta))
        self.state = [self.ant.get_joint_pos(), self.ant.get_position(), self.ant.get_orientation()]
        self.begin_angle = self.state[-1]
        self.begin_pos = self.state[-6:-4]
        self.goal = self._normalize(self.begin_angle + self.delta_theta)
        return self.state, self.delta_theta

    def _is_terminal(self, current):
        if (np.absolute(self._normalize(current) - self.goal) < self.tolerance):
            return True
        else:
            return False

    def step(self, action):
        logging.getLogger("learner").info("stepping")
        self.ant.set_forces_and_trigger(action)
        new_state = [self.ant.get_joint_pos(), self.ant.get_position(), self.ant.get_orientation()]
        net_displacement = new_state[-6:-2] - self.begin_pos
        new_goal = self.goal - self._normalize(new_state[-1])
        if self._is_terminal(new_state[-1]):
            disp_reward = -np.linalg.norm(net_displacement)
            return new_state, new_goal, (disp_reward + self.final_reward), True
        else:
            return new_state, new_goal, self.per_step_reward, self.per_step_reward, False

    def reset(self):
        logging.getLogger("learner").info("resetting")
        self.ant.stop_simulation()
        spawn_rad = np.random.uniform(-self.spawn_radius, +self.spawn_radius)
        angle = np.random.uniform(0, 2*np.pi)
        self.ant.set_position_and_rotation(spawn_rad*np.asarray((np.cos(angle), np.sin(angle)])), np.random.uniform(-np.pi, +np.pi))
        # NOTE : use this only if the stop_simulation() and set_position_and_rotation() doesn't cut it
        # self.ant.init_client()

    def _normalize(self, theta):
        return theta if theta > 0 else return theta + 2*np.pi

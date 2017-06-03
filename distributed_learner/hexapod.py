"""
Interface to control Ant hexapod robot, in one of two modes:
 * Full Force/Torque Control Mode
 * Inverse Kinematics Mode (Hybrid), using Position Control (PID)
"""

import logging
import numpy as np
import thread

import vrep


class Ant(object):
    def __init__(self, server_ip, server_port):
        self.server_ip = server_ip
        self.server_port = server_port
        self.client_id = None
        self.handles = None
        self.body_handle = None
        self.joint_count = None
        self.start_orientation = None
        self.start_pos = None
        self.logger = None

    def init_client(self):
        self.logger = logging.getLogger("learner")
        # Stop any previous simulation
        vrep.simxStopSimulation(-1, vrep.simx_opmode_oneshot_wait)
        # Start a client
        self.client_id = vrep.simxStart(self.server_ip, self.server_port, True, True, 5000, 5)
        if self.client_id == -1:
            self.logger.critical('Failed connecting to remote API server')
            self.logger.critical('EXIT')
            thread.exit()

        # Enable synchronous mode
        e = vrep.simxSynchronous(self.client_id, True)
        self.logger.info("simxSynchronous=%d" % e)
        if e != 0:
            self.logger.critical('Failed enabling remote API synchronous mode')
            self.logger.critical('EXIT')
            thread.exit()

        # Start simulation
        e = vrep.simxStartSimulation(self.client_id, vrep.simx_opmode_blocking)
        self.logger.info("simxStartSimulation=%d" % e)
        if e != 0:
            self.logger.critical('Failed to start simulation')
            self.logger.critical('EXIT')
            thread.exit()

        # Print ping time
        sec, msec = vrep.simxGetPingTime(self.client_id)
        self.logger.info("Started simulation on %s:%d" % (self.server_ip, self.server_port))
        self.logger.info("Ping time: %f" % (sec + msec / 1000.0))

        # Obtain handle for body (for orientation, positions etc)
        _, self.body_handle = vrep.simxGetObjectHandle(self.client_id, 'Ant_body', vrep.simx_opmode_blocking)

        # Obtain joint handles
        _, Ant_joint1Leg1 = vrep.simxGetObjectHandle(self.client_id, 'Ant_joint1Leg1', vrep.simx_opmode_blocking)
        _, Ant_joint2Leg1 = vrep.simxGetObjectHandle(self.client_id, 'Ant_joint2Leg1', vrep.simx_opmode_blocking)
        _, Ant_joint3Leg1 = vrep.simxGetObjectHandle(self.client_id, 'Ant_joint3Leg1', vrep.simx_opmode_blocking)
        _, Ant_joint1Leg2 = vrep.simxGetObjectHandle(self.client_id, 'Ant_joint1Leg2', vrep.simx_opmode_blocking)
        _, Ant_joint2Leg2 = vrep.simxGetObjectHandle(self.client_id, 'Ant_joint2Leg2', vrep.simx_opmode_blocking)
        _, Ant_joint3Leg2 = vrep.simxGetObjectHandle(self.client_id, 'Ant_joint3Leg2', vrep.simx_opmode_blocking)
        _, Ant_joint1Leg3 = vrep.simxGetObjectHandle(self.client_id, 'Ant_joint1Leg3', vrep.simx_opmode_blocking)
        _, Ant_joint2Leg3 = vrep.simxGetObjectHandle(self.client_id, 'Ant_joint2Leg3', vrep.simx_opmode_blocking)
        _, Ant_joint3Leg3 = vrep.simxGetObjectHandle(self.client_id, 'Ant_joint3Leg3', vrep.simx_opmode_blocking)
        _, Ant_joint1Leg4 = vrep.simxGetObjectHandle(self.client_id, 'Ant_joint1Leg4', vrep.simx_opmode_blocking)
        _, Ant_joint2Leg4 = vrep.simxGetObjectHandle(self.client_id, 'Ant_joint2Leg4', vrep.simx_opmode_blocking)
        _, Ant_joint3Leg4 = vrep.simxGetObjectHandle(self.client_id, 'Ant_joint3Leg4', vrep.simx_opmode_blocking)
        _, Ant_joint1Leg5 = vrep.simxGetObjectHandle(self.client_id, 'Ant_joint1Leg5', vrep.simx_opmode_blocking)
        _, Ant_joint2Leg5 = vrep.simxGetObjectHandle(self.client_id, 'Ant_joint2Leg5', vrep.simx_opmode_blocking)
        _, Ant_joint3Leg5 = vrep.simxGetObjectHandle(self.client_id, 'Ant_joint3Leg5', vrep.simx_opmode_blocking)
        _, Ant_joint1Leg6 = vrep.simxGetObjectHandle(self.client_id, 'Ant_joint1Leg6', vrep.simx_opmode_blocking)
        _, Ant_joint2Leg6 = vrep.simxGetObjectHandle(self.client_id, 'Ant_joint2Leg6', vrep.simx_opmode_blocking)
        _, Ant_joint3Leg6 = vrep.simxGetObjectHandle(self.client_id, 'Ant_joint3Leg6', vrep.simx_opmode_blocking)

        # Obtain body joint handles
        _, Ant_neckJoint1 = vrep.simxGetObjectHandle(self.client_id, 'Ant_neckJoint1', vrep.simx_opmode_blocking)
        _, Ant_neckJoint2 = vrep.simxGetObjectHandle(self.client_id, 'Ant_neckJoint2', vrep.simx_opmode_blocking)
        _, Ant_neckJoint3 = vrep.simxGetObjectHandle(self.client_id, 'Ant_neckJoint3', vrep.simx_opmode_blocking)
        _, Ant_leftJawJoint = vrep.simxGetObjectHandle(self.client_id, 'Ant_leftJawJoint', vrep.simx_opmode_blocking)
        _, Ant_rightJawJoint = vrep.simxGetObjectHandle(self.client_id, 'Ant_rightJawJoint', vrep.simx_opmode_blocking)

        # Order : [Ant_joint{1-3}Leg{1-6}, Ant_neckJoint{1-3}, Ant_leftJawJoint, Ant_rightJawJoint]
        self.handles = [Ant_joint1Leg1, Ant_joint2Leg1, Ant_joint3Leg1, Ant_joint1Leg2, Ant_joint2Leg2, Ant_joint3Leg2,
                        Ant_joint1Leg3, Ant_joint2Leg3, Ant_joint3Leg3, Ant_joint1Leg4, Ant_joint2Leg4, Ant_joint3Leg4,
                        Ant_joint1Leg5, Ant_joint2Leg5, Ant_joint3Leg5, Ant_joint1Leg6, Ant_joint2Leg6, Ant_joint3Leg6,
                        Ant_neckJoint1, Ant_neckJoint2, Ant_neckJoint3, Ant_leftJawJoint, Ant_rightJawJoint]

        self.joint_count = len(self.handles)

        # log these for consistency
        self.start_pos = self._get_position()
        self.start_orientation = self._get_orientation()
        pos = self.get_position()
        ori = self.get_orientation()
        jpos = self.get_joint_pos()
        jvel = self.get_joint_vel()
        '''
        correct = False
        while not correct:
            correct = True
            if np.all(self.start_pos == 0):
                correct = False
            if np.all(self.start_orientation == 0):
                correct = False
            if np.all(jpos == 0):
                correct = False
            time.sleep(0.1)
        logging.getLogger("learner").info("started sim")
        
        # Configure them to have almost infinite target velocity
        # Use only if in pure Force/Torque mode without position control (PID)
        _ = vrep.simxPauseCommunication(self.client_id, True);
        for joint_handle in self.handles:
            _ = vrep.simxSetJointTargetVelocity(self.client_id, joint_handle, 9999.0, vrep.simx_opmode_blocking)
        _ = vrep.simxPauseCommunication(self.client_id, False);
        '''

    # force_vec = [Ant_joint{1-3}Leg{1-6}, Ant_neckJoint{1-3}, Ant_leftJawJoint, Ant_rightJawJoint]
    def set_forces(self, force_vec):
        forces = force_vec.flatten().tolist()
        _ = vrep.simxPauseCommunication(self.client_id, True)
        for handle_idx in range(0, self.joint_count):
            _ = vrep.simxSetJointForce(self.client_id, self.handles[handle_idx], forces[handle_idx],
                                       vrep.simx_opmode_oneshot)
        _ = vrep.simxPauseCommunication(self.client_id, False)

    # set force_vec and trigger simulation step
    def set_forces_and_trigger(self, force_vec):
        self.set_forces(force_vec)
        _ = vrep.simxSynchronousTrigger(self.client_id)

    # Get joint positions
    def get_joint_pos(self):
        jpos = np.zeros((1, self.joint_count))
        for handle_idx in range(0, self.joint_count):
            _, jpos[0, handle_idx] = vrep.simxGetJointPosition(self.client_id, self.handles[handle_idx],
                                                               vrep.simx_opmode_streaming)
        return jpos

    # Get joint velocities
    def get_joint_vel(self):
        jvel = np.zeros((1, self.joint_count))
        for handle_idx in range(0, self.joint_count):
            _, jvel[0, handle_idx] = vrep.simxGetObjectFloatParameter(self.client_id, self.handles[handle_idx], 2012,
                                                                      vrep.simx_opmode_streaming)
        return jvel

    # returns euler orientation (alpha, beta, gamma) for the body
    def get_orientation(self):
        _, euler_angles = vrep.simxGetObjectOrientation(self.client_id, self.body_handle, -1,
                                                        vrep.simx_opmode_streaming)
        return np.asarray((euler_angles[0], euler_angles[1], euler_angles[2])).reshape((1, -1))

    # returns Cartesian coordinates (x,y,z) for the body
    def get_position(self):
        _, position = vrep.simxGetObjectPosition(self.client_id, self.body_handle, -1, vrep.simx_opmode_streaming)
        return np.asarray((position[0], position[1], position[2])).reshape((1, -1))

    # copy functions using blocking mode instead of streaming to get correct values
    def _get_orientation(self):
        _, euler_angles = vrep.simxGetObjectOrientation(self.client_id, self.body_handle, -1, vrep.simx_opmode_blocking)
        return np.asarray((euler_angles[0], euler_angles[1], euler_angles[2])).reshape((1, -1))

    # returns Cartesian coordinates (x,y,z) for the body
    def _get_position(self):
        _, position = vrep.simxGetObjectPosition(self.client_id, self.body_handle, -1, vrep.simx_opmode_blocking)
        return np.asarray((position[0], position[1], position[2])).reshape((1, -1))

    # set (x,y) position and rotation about z-axis
    def set_position_and_rotation(self, position, rotation):
        # time.sleep(0.2)
        empty_buff = bytearray()
        args = [position[0], position[1], self.start_pos[0, 2], self.start_orientation[0, 0],
                self.start_orientation[0, 1],
                rotation]
        self.logger.info("args:%s" % str(args))
        res, ret_ints, ret_floats, ret_strings, ret_buffer = vrep.simxCallScriptFunction(self.client_id, 'Ant',
                                                                                         vrep.sim_scripttype_childscript,
                                                                                         'resetScene', [], args, [],
                                                                                         empty_buff,
                                                                                         vrep.simx_opmode_oneshot_wait)
        if res != vrep.simx_return_ok:
            self.logger.critical("failed to set position and orientation")

    # Set custom dt for simulation
    def set_custom_dt(self, custom_dt):
        _ = vrep.simxSetFloatingParameter(self.client_id, vrep.sim_floatparam_simulation_time_step, custom_dt,
                                          vrep.simx_opmode_blocking)

    def start_simulation(self):
        # Start simulation
        e = vrep.simxStartSimulation(self.client_id, vrep.simx_opmode_blocking)
        if e != 0:
            self.logger.critical('Failed to start simulation')

        # Print ping time
        sec, msec = vrep.simxGetPingTime(self.client_id)
        self.logger.info("Started simulation on %s:%d" % (self.server_ip, self.server_port))
        self.logger.info("Ping time: %f" % (sec + msec / 1000.0))

        # required
        self.start_pos = self._get_position()
        self.start_orientation = self._get_orientation()
        jpos = self.get_joint_pos()
        jvel = self.get_joint_vel()
        '''
        correct = False
        while not correct:
            correct = True
            if np.all(self.start_pos == 0):
                correct = False
            if np.all(self.start_orientation == 0):
                correct = False
            if np.all(jpos == 0):
                correct = False
            time.sleep(0.1)
        logging.getLogger("learner").info("started sim")
        '''

    def stop_simulation(self):
        # issue command to stop simulation
        vrep.simxStopSimulation(self.client_id, vrep.simx_opmode_blocking)
        # stop all streaming
        for handle_idx in range(0, self.joint_count):
            _, _ = vrep.simxGetJointPosition(self.client_id, self.handles[handle_idx], vrep.simx_opmode_discontinue)

        for handle_idx in range(0, self.joint_count):
            _, _ = vrep.simxGetObjectFloatParameter(self.client_id, self.handles[handle_idx], 2012,
                                                    vrep.simx_opmode_discontinue)

        _, _ = vrep.simxGetObjectPosition(self.client_id, self.body_handle, -1, vrep.simx_opmode_discontinue)
        _, _ = vrep.simxGetObjectOrientation(self.client_id, self.body_handle, -1, vrep.simx_opmode_discontinue)
        # Just to make sure this gets executed
        vrep.simxGetPingTime(self.client_id)
        self.logger.info("simulation stopped")

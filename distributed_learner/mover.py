"""
Interface to control a simple 2 wheeled robot:
 * Inverse Kinematics Mode (Hybrid), using TargetVelocity
"""

import logging
import numpy as np
import thread

import vrep


class Mover(object):
    def __init__(self, server_ip, server_port):
        self.server_ip = server_ip
        self.server_port = server_port
        self.client_id = None
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
        _, self.reference = vrep.simxGetObjectHandle(self.client_id, 'ResizableFloor_5_25', vrep.simx_opmode_blocking)

        # Obtain handle for body (for orientation, positions etc)
        _, self.body_handle = vrep.simxGetObjectHandle(self.client_id, 'main', vrep.simx_opmode_blocking)

        # Obtain joint handles
        _, self.left_joint = vrep.simxGetObjectHandle(self.client_id, 'dr12_leftJoint_', vrep.simx_opmode_blocking)
        _, self.right_joint = vrep.simxGetObjectHandle(self.client_id, 'dr12_rightJoint_', vrep.simx_opmode_blocking)

        # obtain force sensors
        _, force_sensor = vrep.simxGetObjectHandle(self.client_id, 'dr12_bumperForceSensor_', vrep.simx_opmode_blocking)

        self.joint_count = 2
        self.signal_values()
        self.wait_till_stream()

        # log these for consistency
        self.start_pos = self.get_position()
        self.start_orientation = self.get_orientation()

    def set_target_vel(self, vel_vec):
        _ = vrep.simxSetJointTargetVelocity(self.client_id, self.left_joint, vel_vec[0].item(),
                                            vrep.simx_opmode_oneshot)
        _ = vrep.simxSetJointTargetVelocity(self.client_id, self.right_joint, vel_vec[1].item(),
                                            vrep.simx_opmode_oneshot)
        _ = vrep.simxSynchronousTrigger(self.client_id)

    # Get joint positions
    def get_joint_pos(self):
        jpos = np.zeros((1, 2))
        _, jpos[0, 0] = vrep.simxGetJointPosition(self.client_id, self.left_joint, vrep.simx_opmode_buffer)
        _, jpos[0, 1] = vrep.simxGetJointPosition(self.client_id, self.right_joint, vrep.simx_opmode_buffer)
        return jpos

    # Get joint velocities
    def get_joint_vel(self):
        jvel = np.zeros((1, 2))
        _, jvel[0, 0] = vrep.simxGetObjectFloatParameter(self.client_id, self.left_joint, 2012, vrep.simx_opmode_buffer)
        _, jvel[0, 1] = vrep.simxGetObjectFloatParameter(self.client_id, self.right_joint, 2012,
                                                         vrep.simx_opmode_buffer)
        return jvel

    # returns euler orientation (alpha, beta, gamma) for the body
    def get_orientation(self):
        _, euler_angles = vrep.simxGetObjectOrientation(self.client_id, self.body_handle, -1, vrep.simx_opmode_buffer)
        return np.asarray((euler_angles[0], euler_angles[1], euler_angles[2])).reshape((1, -1))

    # returns Cartesian coordinates (x,y,z) for the body
    def get_position(self):
        _, position = vrep.simxGetObjectPosition(self.client_id, self.body_handle, -1, vrep.simx_opmode_buffer)
        return np.asarray((position[0], position[1], position[2])).reshape((1, -1))

    # set (x,y) position and rotation about z-axis
    def set_position_and_rotation(self, position, rotation):
        # time.sleep(0.2)
        empty_buff = bytearray()
        # self.start_pos = self.get_position()
        # self.start_orientation = self.get_orientation()
        args = [position[0].item(), position[1].item(), self.start_pos[0, 2].item(), rotation]
        # self.logger.info("args:%s" % str(args))
        emptyBuff = bytearray()
        res, ret_ints, ret_floats, ret_strings, ret_buffer = vrep.simxCallScriptFunction(self.client_id, 'main',
                                                                                         vrep.sim_scripttype_childscript,
                                                                                         'resetScene', [], args, [],
                                                                                         empty_buff,
                                                                                         vrep.simx_opmode_blocking)
        if res != vrep.simx_return_ok:
            self.logger.critical("failed to set position and orientation")

    # Set custom dt for simulation
    def set_custom_dt(self, custom_dt):
        _ = vrep.simxSetFloatingParameter(self.client_id, vrep.sim_floatparam_simulation_time_step, custom_dt,
                                          vrep.simx_opmode_blocking)

    def signal_values(self):
        vrep.simxGetObjectOrientation(self.client_id, self.body_handle, -1, vrep.simx_opmode_streaming)
        vrep.simxGetObjectPosition(self.client_id, self.body_handle, -1, vrep.simx_opmode_streaming)
        _, _ = vrep.simxGetJointPosition(self.client_id, self.left_joint, vrep.simx_opmode_streaming)
        _, _ = vrep.simxGetJointPosition(self.client_id, self.right_joint, vrep.simx_opmode_streaming)
        _, _ = vrep.simxGetObjectFloatParameter(self.client_id, self.left_joint, 2012, vrep.simx_opmode_streaming)
        _, _ = vrep.simxGetObjectFloatParameter(self.client_id, self.right_joint, 2012, vrep.simx_opmode_streaming)

    def wait_till_stream(self):
        ret = vrep.simx_return_illegal_opmode_flag
        while ret != vrep.simx_return_ok:
            ret, _ = vrep.simxGetObjectOrientation(self.client_id, self.body_handle, -1, vrep.simx_opmode_buffer)

        ret = vrep.simx_return_illegal_opmode_flag
        while ret != vrep.simx_return_ok:
            ret, _ = vrep.simxGetObjectPosition(self.client_id, self.body_handle, -1, vrep.simx_opmode_buffer)

        ret = vrep.simx_return_illegal_opmode_flag
        while ret != vrep.simx_return_ok:
            ret, _ = vrep.simxGetJointPosition(self.client_id, self.left_joint, vrep.simx_opmode_buffer)

        ret = vrep.simx_return_illegal_opmode_flag
        while ret != vrep.simx_return_ok:
            ret, _ = vrep.simxGetJointPosition(self.client_id, self.right_joint, vrep.simx_opmode_buffer)

        ret = vrep.simx_return_illegal_opmode_flag
        while ret != vrep.simx_return_ok:
            ret, _ = vrep.simxGetObjectFloatParameter(self.client_id, self.right_joint, 2012, vrep.simx_opmode_buffer)

        ret = vrep.simx_return_illegal_opmode_flag
        while ret != vrep.simx_return_ok:
            ret, _ = vrep.simxGetObjectFloatParameter(self.client_id, self.right_joint, 2012, vrep.simx_opmode_buffer)

    def start_simulation(self):
        # Start simulation
        e = vrep.simxStartSimulation(self.client_id, vrep.simx_opmode_blocking)
        if e != 0:
            self.logger.critical('Failed to start simulation')

        # Print ping time
        sec, msec = vrep.simxGetPingTime(self.client_id)
        self.logger.info("Started simulation on %s:%d" % (self.server_ip, self.server_port))
        self.logger.info("Ping time: %f" % (sec + msec / 1000.0))

        # block till stream
        self.signal_values()
        self.wait_till_stream()

        # required
        self.start_pos = self.get_position()
        self.start_orientation = self.get_orientation()

    def stop_simulation(self):
        # issue command to stop simulation
        vrep.simxStopSimulation(self.client_id, vrep.simx_opmode_blocking)
        # stop all streaming
        vrep.simxGetObjectOrientation(self.client_id, self.body_handle, -1, vrep.simx_opmode_discontinue)
        vrep.simxGetObjectPosition(self.client_id, self.body_handle, -1, vrep.simx_opmode_discontinue)
        _, _ = vrep.simxGetJointPosition(self.client_id, self.left_joint, vrep.simx_opmode_discontinue)
        _, _ = vrep.simxGetJointPosition(self.client_id, self.right_joint, vrep.simx_opmode_discontinue)
        _, _ = vrep.simxGetObjectFloatParameter(self.client_id, self.left_joint, 2012, vrep.simx_opmode_discontinue)
        _, _ = vrep.simxGetObjectFloatParameter(self.client_id, self.right_joint, 2012, vrep.simx_opmode_discontinue)
        # Just to make sure this gets executed
        vrep.simxGetPingTime(self.client_id)
        self.logger.info("simulation stopped")

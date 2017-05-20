"""
Interface to expose V-REP remote API for Asti humanoid robot, in one of two modes:
 * Full Force/Torque Control Mode
 * Inverse Kinematics Mode (Hybrid), using Position Control (PID)
"""

import time
import numpy as np
import vrep
import math

class Asti:
    def __init__(server_ip, server_port):
        self.server_ip = server_ip
        self.server_port = server_port
        self.joint_count = 20
        self.client_id = None
        self.handles = None

    def init_client():
        # Close any open comms
        _ = vrep.simxStopSimulation(client_id, vrep.simx_opmode_blocking)

        # Start a new client
        self.client_id = vrep.simxStart(self.server_ip, self.server_port, True, True, 5000, 5)
        assert self.client_id != -1, 'Failed connecting to remote API server'

        # Enable synchronous mode
        e = vrep.simxSynchronous(self.client_id, True)
        assert e != -1, 'Failed enabling remote API synchronous mode'

        # Start simulation
        e = vrep.simxStartSimulation(self.client_id, vrep.simx_opmode_blocking)
        assert e != -1, 'Failed to start simulation'

        # Print ping time
        sec, msec = vrep.simxGetPingTime(self.client_id)
        print "Started simulation on %s:%d" % (self.server_ip, self.server_port)
        print "Ping time: %f" % (sec + msec / 1000.0)

        # Obtain handle for body (for orientation, positions etc)
        _, self.body_handle = vrep.simxGetObjectHandle(self.client_id, '_asti_body', vrep.simx_opmode_blocking)

        # Obtain joint handles
        _, neckJoint0 = vrep.simxGetObjectHandle(self.client_id, 'neckJoint0', vrep.simx_opmode_blocking)
        _, neckJoint1 = vrep.simxGetObjectHandle(self.client_id, 'neckJoint1', vrep.simx_opmode_blocking)
        neckJoints = [neckJoint0, neckJoint1]
        _, leftArmJoint0 = vrep.simxGetObjectHandle(self.client_id, 'leftArmJoint0', vrep.simx_opmode_blocking)
        _, leftArmJoint1 = vrep.simxGetObjectHandle(self.client_id, 'leftArmJoint1', vrep.simx_opmode_blocking)
        _, leftArmJoint2 = vrep.simxGetObjectHandle(self.client_id, 'leftArmJoint2', vrep.simx_opmode_blocking)
        leftArmJoints = [leftArmJoint0, leftArmJoint1, leftArmJoint2]
        _, rightArmJoint0 = vrep.simxGetObjectHandle(self.client_id, 'rightArmJoint0', vrep.simx_opmode_blocking)
        _, rightArmJoint1 = vrep.simxGetObjectHandle(self.client_id, 'rightArmJoint1', vrep.simx_opmode_blocking)
        _, rightArmJoint2 = vrep.simxGetObjectHandle(self.client_id, 'rightArmJoint2', vrep.simx_opmode_blocking)
        rightArmJoints = [rightArmJoint0, rightArmJoint1, rightArmJoint2]
        _, leftLegJoint0 = vrep.simxGetObjectHandle(self.client_id, 'leftLegJoint0', vrep.simx_opmode_blocking)
        _, leftLegJoint1 = vrep.simxGetObjectHandle(self.client_id, 'leftLegJoint1', vrep.simx_opmode_blocking)
        _, leftLegJoint2 = vrep.simxGetObjectHandle(self.client_id, 'leftLegJoint2', vrep.simx_opmode_blocking)
        _, leftLegJoint3 = vrep.simxGetObjectHandle(self.client_id, 'leftLegJoint3', vrep.simx_opmode_blocking)
        _, leftLegJoint4 = vrep.simxGetObjectHandle(self.client_id, 'leftLegJoint4', vrep.simx_opmode_blocking)
        _, leftLegJoint5 = vrep.simxGetObjectHandle(self.client_id, 'leftLegJoint5', vrep.simx_opmode_blocking)
        leftLegJoints = [leftLegJoint0, leftLegJoint1, leftLegJoint2, leftLegJoint3, leftLegJoint4, leftLegJoint5]
        _, rightLegJoint0 = vrep.simxGetObjectHandle(self.client_id, 'rightLegJoint0', vrep.simx_opmode_blocking)
        _, rightLegJoint1 = vrep.simxGetObjectHandle(self.client_id, 'rightLegJoint1', vrep.simx_opmode_blocking)
        _, rightLegJoint2 = vrep.simxGetObjectHandle(self.client_id, 'rightLegJoint2', vrep.simx_opmode_blocking)
        _, rightLegJoint3 = vrep.simxGetObjectHandle(self.client_id, 'rightLegJoint3', vrep.simx_opmode_blocking)
        _, rightLegJoint4 = vrep.simxGetObjectHandle(self.client_id, 'rightLegJoint4', vrep.simx_opmode_blocking)
        _, rightLegJoint5 = vrep.simxGetObjectHandle(self.client_id, 'rightLegJoint5', vrep.simx_opmode_blocking)
        rightLegJoints = [rightLegJoint0, rightLegJoint1, rightLegJoint2, rightLegJoint3, rightLegJoint4, rightLegJoint5]
        self.handles = [joint_handle for joint_handle_cluster in [neckJoints, leftArmJoints, rightArmJoints, leftLegJoints, rightLegJoints] for joint_handle in joint_handle_cluster]

        '''
        # Configure them to have almost infinite target velocity
        # Use only if in pure Force/Torque mode without position control (PID)
        _ = vrep.simxPauseCommunication(self.client_id, True);
        for joint_handle in self.handles:
            _ = vrep.simxSetJointTargetVelocity(self.client_id, joint_handle, 9999.0, vrep.simx_opmode_blocking)
        _ = vrep.simxPauseCommunication(self.client_id, False);
        '''

    # force_vec = [neckJoints, leftArmJoints, rightArmJoints, leftLegJoints, rightLegJoints]
    def set_forces(force_vec):
        _ = vrep.simxPauseCommunication(self.client_id, True);
        for handle_idx in range(0, self.joint_count):
            _ = vrep.simxSetJointForce(self.client_id, self.handles[handle_idx], force_vec[handle_idx], vrep.simx_opmode_blocking)
        _ = vrep.simxPauseCommunication(self.client_id, False);

    # set force_vec and trigger simulation step
    def set_forces_and_trigger(force_vec):
        self.set_forces(force_vec)
        _ = vrep.simxSynchronousTrigger(self.client_id);

    # Get joint positions
    def get_joint_pos():
        jpos = np.zeros(self.joint_count);
        for handle_idx in range(0, self.joint_count):
            _, jpos[handle_idx] = vrep.simxGetJointPosition(self.client_id, self.handles[handle_idx], vrep.simx_opmode_blocking)

    # returns euler orientation (alpha, beta, gamma) for the body
    def get_orientation():
        _, euler_angles = vrep.simxGetObjectOrientation(self.client_id, self.body_handle, vrep.simx_opmode_blocking)
        return euler_angles[0], euler_angles[1], euler_angles[2]

    # returns Cartesian coordinates (x,y,z) for the body
    def get_position():
        _, position = vrep.simxGetObjectPosition(self.client_id, self.body_handle, vrep.simx_opmode_blocking)
        return position[0], position[1], position[2]

    # Set custom dt for simulation
    def set_custom_dt(custom_dt):
        _ = vrep.simxSetFloatingParameter(self.client_id, vrep.sim_floatparam_simulation_time_step, custom_dt, vrep.simx_opmode_blocking)

    def stop_simulation():
        _ = vrep.simxStopSimulation(self.client_id, vrep.simx_opmode_blocking)

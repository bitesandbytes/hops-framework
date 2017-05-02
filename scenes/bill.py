import time
import numpy as np
import vrep
import math

# Bill theta vector dims
theta_dim = 10;

class PlainBillDynamics:
    def __init__(self):
        # Define theta, omega and alpha arrays
        self.theta = np.zeros(theta_dim);
        self.omega = np.zeros(theta_dim);
        self.alpha = np.zeros(theta_dim);

        # Delta between two updates
        self.time_delta = 0.005;

    def apply_alpha(alpha_vec):
        omega_new = self.omega + 0.5*self.time_delta*(self.alpha + alpha_vec);
        self.omega += self.time_delta*self.alpha;
        self.theta += self.omega*self.time_delta + self.alpha


#import contexttimer

if __name__ == '__main__':
    try:
        client_id
    except NameError:
        client_id = -1
    e = vrep.simxStopSimulation(client_id, vrep.simx_opmode_oneshot_wait)
    vrep.simxFinish(-1)
    client_id = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)

    assert client_id != -1, 'Failed connecting to remote API server'

    e = vrep.simxStartSimulation(client_id, vrep.simx_opmode_oneshot_wait)

    # print ping time
    sec, msec = vrep.simxGetPingTime(client_id)
    print "Ping time: %f" % (sec + msec / 1000.0)

    # Handle of neck joint
    e, neck = vrep.simxGetObjectHandle(client_id, 'Joint10#0', vrep.simx_opmode_oneshot_wait)

    e, neck_pos = vrep.simxGetJointPosition(client_id, neck, vrep.simx_opmode_streaming)

    print "Current position = %0.3f degrees" % neck_pos
    e = vrep.simxSetJointPosition(client_id, neck, 0.5, vrep.simx_opmode_streaming)

    e, neck_pos = vrep.simxGetJointPosition(client_id, neck, vrep.simx_opmode_streaming)

    print "Current position = %0.3f degrees" % neck_pos

    # Sleep for a small time
    for i in range(0,1800):
        #time.sleep(0.1)
        print "setting current position as %d" % math.fmod(i,360)

        e = vrep.simxSetJointPosition(client_id, neck, i/180. * math.pi, vrep.simx_opmode_oneshot_wait)
        e, neck_pos = vrep.simxGetJointPosition(client_id, neck, vrep.simx_opmode_oneshot_wait)

        print "Current position = %0.3f degrees" % (neck_pos/math.pi * 180)

    # Sleep for a small time.
    time.sleep(0.3)

    e = vrep.simxStopSimulation(client_id, vrep.simx_opmode_oneshot_wait)

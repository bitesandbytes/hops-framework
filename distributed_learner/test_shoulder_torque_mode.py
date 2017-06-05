import math
import time

import vrep

# import contexttimer

if __name__ == '__main__':
    try:
        client_id
    except NameError:
        client_id = -1
    e = vrep.simxStopSimulation(client_id, vrep.simx_opmode_blocking)

    # Close any open comms
    vrep.simxFinish(-1)

    # Start a new client
    client_id = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)
    assert client_id != -1, 'Failed connecting to remote API server'

    # Enable synchronous triggers
    e = vrep.simxSynchronous(client_id, True)

    # Start simulation
    e = vrep.simxStartSimulation(client_id, vrep.simx_opmode_blocking)

    # print ping time
    sec, msec = vrep.simxGetPingTime(client_id)
    print "Ping time: %f" % (sec + msec / 1000.0)

    # Obtain joint handle
    e, joint = vrep.simxGetObjectHandle(client_id, 'neckJoint1', vrep.simx_opmode_blocking)

    # Obtain initial pos
    e, jpos = vrep.simxGetJointPosition(client_id, joint, vrep.simx_opmode_blocking)
    print "Current position = %0.3f degrees" % ((jpos * 1.0) / math.pi * 180.0)

    # Set custom dt
    # _ = vrep.simxSetFloatingParameter(client_id, vrep.sim_floatparam_simulation_time_step, 0.01, vrep.simx_opmode_blocking)

    # Synchronous
    for i in range(0, 1000):

        # Compute and Set torque
        e = vrep.simxSetJointTargetVelocity(client_id, joint, 9999.0, vrep.simx_opmode_blocking);
        if e != 0: raise Exception()
        e = vrep.simxSetJointForce(client_id, joint, 1, vrep.simx_opmode_blocking)
        if e != 0: raise Exception()
        # e = vrep.simxSetJointPosition(client_id, joint, i/180. * math.pi, vrep.simx_opmode_blocking)

        # Send Synchronous trigger
        e = vrep.simxSynchronousTrigger(client_id)

        # Wait for exec to complete
        # e = vrep.simxGetPingTime(client_id)

        # Obtain position
        e, jpos = vrep.simxGetJointPosition(client_id, joint, vrep.simx_opmode_blocking)
        print "Current position = %0.3f degrees" % ((jpos * 1.0) / math.pi * 180.0)

    # Sleep for a small time.
    time.sleep(3)

    e = vrep.simxStopSimulation(client_id, vrep.simx_opmode_blocking)

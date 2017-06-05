import numpy as np
import time

import vrep

# import contexttimer

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

    _, ant = vrep.simxGetObjectHandle(client_id, 'Ant', vrep.simx_opmode_oneshot_wait)
    _, pos = vrep.simxGetObjectPosition(client_id, ant, -1, vrep.simx_opmode_blocking)
    _, orient = vrep.simxGetObjectOrientation(client_id, ant, -1, vrep.simx_opmode_blocking)
    print "Start position : %f, %f, %f" % (pos[0], pos[1], pos[2])
    print "Start orientation : %f, %f, %f" % (orient[0], orient[1], orient[2])
    # Change position and orientation
    new_pos = pos
    new_pos[0] = np.random.uniform(-1.5, +1.5)
    new_pos[1] = np.random.uniform(-1.5, +1.5)
    new_orient = orient
    # new_orient[1] = np.random.uniform(-1.57, +1.57)
    new_orient[2] = +3.14 * 1.5
    '''
    _ = vrep.simxSetObjectPosition(client_id, ant, -1, new_pos, vrep.simx_opmode_blocking)
    _ = vrep.simxSetObjectOrientation(client_id, ant, -1, new_orient, vrep.simx_opmode_blocking)
    '''
    print "Setting position : %f, %f, %f" % (new_pos[0], new_pos[1], new_pos[2])
    print "Setting orientation : %f, %f, %f" % (new_orient[0], new_orient[1], new_orient[2])
    args = [new_pos[0], new_pos[1], new_pos[2], new_orient[0], new_orient[1], new_orient[2]]
    emptyBuff = bytearray()
    res, retInts, retFloats, retStrings, retBuffer = vrep.simxCallScriptFunction(client_id, 'Ant',
                                                                                 vrep.sim_scripttype_childscript,
                                                                                 'resetScene', [], args, [], emptyBuff,
                                                                                 vrep.simx_opmode_blocking)
    if res == vrep.simx_return_ok:
        print ('Worked')  # display the reply from V-REP (in this case, the handle of the created dummy)
    else:
        print ('Failed')

    _, pos = vrep.simxGetObjectPosition(client_id, ant, -1, vrep.simx_opmode_blocking)
    _, orient = vrep.simxGetObjectOrientation(client_id, ant, -1, vrep.simx_opmode_blocking)
    print "Final position : %f, %f, %f" % (pos[0], pos[1], pos[2])
    print "Final orientation : %f, %f, %f" % (orient[0], orient[1], orient[2])
    time.sleep(10)

    '''
    # Handle of neck joint
    e, neck = vrep.simxGetObjectHandle(client_id, 'Bill_neck', vrep.simx_opmode_oneshot_wait)
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
    '''
    e = vrep.simxStopSimulation(client_id, vrep.simx_opmode_oneshot_wait)

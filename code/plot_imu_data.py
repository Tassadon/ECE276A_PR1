
import matplotlib.pyplot as plt
import autograd.numpy as np
import transforms3d

def plot(dataset="1",path="../numpy files"):
    quarternions_T = np.load(path + "/data_set_" + dataset + "_raw_quarternions.npy")
    optimized_qts = np.load(path + "/data_set_" + dataset + "_optimized_quarternions.npy")
    rots = np.load(path + "/data_set_" + dataset + "_vicon_data.npy")

    euler_angles = list(map(lambda quart: transforms3d.euler.quat2euler(quart),quarternions_T))
    roll = list(map(lambda angles: angles[0],euler_angles))
    pitch = list(map(lambda angles: angles[1],euler_angles))
    yaw = list(map(lambda angles: angles[2],euler_angles))

    opt_euler_angles = list(map(lambda quart: transforms3d.euler.quat2euler(quart),optimized_qts))
    opt_roll = list(map(lambda angles: angles[0],opt_euler_angles))
    opt_pitch = list(map(lambda angles: angles[1],opt_euler_angles))
    opt_yaw = list(map(lambda angles: angles[2],opt_euler_angles))

    fig, ax = plt.subplots(3,1,figsize=(6,6))
    ax[0].plot(roll)
    ax[0].set_title("roll in radians")
    ax[0].plot(opt_roll)
    ax[0].plot(rots[:,0])
    ax[1].plot(pitch)
    ax[1].set_title("yaw in radians",x=.2,y=0)
    ax[1].plot(opt_pitch)
    ax[1].plot(rots[:,1])
    ax[2].plot(yaw)
    ax[2].plot(opt_yaw)
    ax[2].set_title("yaw in radians")
    ax[2].plot(rots[:,2])
    ax[0].legend(['raw data','optimized data','VICON data'],loc='lower right')
    ax[1].legend(['raw data','optimized data','VICON data'],loc='lower right')
    ax[2].legend(['raw data','optimized data','VICON data'],loc='lower right')
    fig.suptitle('Yaw Pitch and Roll calculations from dataset: ' + dataset)
    plt.show()

if __name__ == "__main__":
    plot(dataset="2")
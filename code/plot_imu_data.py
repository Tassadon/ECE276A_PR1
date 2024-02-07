
import matplotlib.pyplot as plt
import autograd.numpy as np
import transforms3d

def plot(dataset="1",path="../numpy files",testing=False):
    quarternions_T = np.load(path + "/data_set_" + dataset + "_raw_quarternions.npy")
    optimized_qts = np.load(path + "/data_set_" + dataset + "_optimized_quarternions.npy")
    if not testing:
        rots = np.load(path + "/data_set_" + dataset + "_vicon_data.npy")

    euler_angles = list(map(lambda quart: transforms3d.euler.quat2euler(quart),quarternions_T))
    roll = list(map(lambda angles: angles[0],euler_angles))
    pitch = list(map(lambda angles: angles[1],euler_angles))
    yaw = list(map(lambda angles: angles[2],euler_angles))

    opt_euler_angles = list(map(lambda quart: transforms3d.euler.quat2euler(quart),optimized_qts))
    opt_roll = list(map(lambda angles: angles[0],opt_euler_angles))
    opt_pitch = list(map(lambda angles: angles[1],opt_euler_angles))
    opt_yaw = list(map(lambda angles: angles[2],opt_euler_angles))

    fig, ax = plt.subplots(3,1,figsize=(10,10))
    ax[0].plot(roll)
    ax[0].set_title("roll in radians")
    ax[0].plot(opt_roll)
    ax[1].plot(pitch)
    ax[1].set_title("yaw in radians")
    ax[1].plot(opt_pitch)
    ax[2].plot(yaw)
    ax[2].plot(opt_yaw)
    ax[2].set_title("yaw in radians")
    if not testing:
        ax[0].plot(rots[:,0])
        ax[1].plot(rots[:,1])
        ax[2].plot(rots[:,2])
        ax[0].legend(['raw data','optimized data','VICON data'],loc='lower right')
        ax[1].legend(['raw data','optimized data','VICON data'],loc='lower right')
        ax[2].legend(['raw data','optimized data','VICON data'],loc='lower right')
    if testing:
        ax[0].legend(['raw data','optimized data'],loc='lower right')
        ax[1].legend(['raw data','optimized data'],loc='lower right')
        ax[2].legend(['raw data','optimized data'],loc='lower right')
        
    fig.suptitle('Yaw Pitch and Roll calculations from dataset: ' + dataset)
    plt.show()

if __name__ == "__main__":
    dataset_given = False
    while(not dataset_given):
        dataset = input("Please enter the dataset you want to graph: ")
        if dataset not in ["1","2","3","4","5","6","7","8","9","10","11"]:
            print("please enter a valid dataset")
        else:
            dataset_given = True
    
    plot(dataset=dataset)
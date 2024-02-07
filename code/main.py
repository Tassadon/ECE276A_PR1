import load_data, rotplot
import math
import matplotlib.pyplot as plt
import autograd.numpy as np
import transforms3d
import optimization
import quarternions

def generate_and_save_optimization(dataset="1",epochs=10,alpha=.003,testing=False):

    [camera,imu,Vicd] = load_data.load(fname="../../trainset",dataset=dataset,testing=testing)
    rots = []
    if not testing:
        for i in range(Vicd['rots'].shape[2]):
            rots.append(list(transforms3d.euler.mat2euler(Vicd['rots'][:,:,i])))

        rots = np.array(rots)
    sc_ft_ang=3300/1023*(math.pi)/180/3.33
    sc_ft_acc=3300/1023/300

    Wz = imu['vals'][3]
    Wx = imu['vals'][4]
    Wy = imu['vals'][5]
    Ang_Veloc = np.array([Wx*sc_ft_ang,Wy*sc_ft_ang,Wz*sc_ft_ang])

    Ax = imu['vals'][0].astype(np.float32)*sc_ft_acc
    Ay = imu['vals'][1].astype(np.float32)*sc_ft_acc
    Az = imu['vals'][2].astype(np.float32)*sc_ft_acc
    Ang_Acc = np.array([-Ax,-Ay,Az],dtype=np.float32)

    time = imu['ts'][0]

    tau = np.zeros((len(time)-1,1))
    for i,t in enumerate(time,start=0):
        if i == 0:
            continue
        else:
            tau[i-1] = t - time[i-1]
    bias_ang_x=np.average(Ang_Veloc[0,0:100])
    bias_ang_y=np.average(Ang_Veloc[1,0:100])
    bias_ang_z=np.average(Ang_Veloc[2,0:100])
    bias_acc_x=np.average(Ang_Acc[0,0:100])
    bias_acc_y=np.average(Ang_Acc[1,0:100])
    bias_acc_z=np.average(Ang_Acc[2,0:100])
    Ang_Veloc[0] = Ang_Veloc[0] - bias_ang_x
    Ang_Veloc[1] = Ang_Veloc[1] - bias_ang_y
    Ang_Veloc[2] = Ang_Veloc[2] - bias_ang_z
    Ang_Acc[0] = Ang_Acc[0] - bias_acc_x
    Ang_Acc[1] = Ang_Acc[1] - bias_acc_y
    Ang_Acc[2] = Ang_Acc[2] - bias_acc_z + 1

    quarternions_T = quarternions.predict_next_quarternion(tau,Ang_Veloc)
    quarts = np.array(quarternions_T)
    optimized_qts = optimization.optimize(quarts,Ang_Acc.T,tau[:,0],Ang_Veloc.T,alpha=alpha,epochs=epochs)
    np.save("../numpy files/data_set_" + dataset + "_optimized_quarternions",optimized_qts)
    np.save("../numpy files/data_set_" + dataset + "_raw_quarternions",quarternions_T)
    if not testing:
        np.save("../numpy files/data_set_" + dataset + "_vicon_data",rots)

if __name__ == "__main__":
    dataset_list=["1","2","3","4","5","6","7","8","9","10","11"]
    for dataset in dataset_list:
        if dataset == "10" or dataset == "11":
            generate_and_save_optimization(dataset=dataset,epochs=10,alpha=.05,testing=False)
        else:
            generate_and_save_optimization(dataset=dataset,epochs=10,alpha=.05,testing=True)
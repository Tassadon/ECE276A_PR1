import load_data
import autograd.numpy as np
import transforms3d
import matplotlib.pyplot as plt
from tqdm import tqdm

def rotate_cartesian_coordinates(x):

    return [np.cos(x[1])*np.cos(x[0]),-np.cos(x[1])*np.sin(x[0]),-np.sin(x[1])]

def rotate_spherical_coordinates(x):
    return [np.arcsin(-x[2]/np.linalg.norm(x)),np.arctan2(x[1],x[0]),np.linalg.norm(x)]

def get_image_plane_coordinates(x):
    return np.array([(x[0]+(np.pi/2))*(960/np.pi),(x[1]+(np.pi))*(1280/(2*np.pi))]).astype(int)


def spherical_to_cartesian_coordinates(x):
    return [np.cos(x[1])*np.cos(x[0]),-np.cos(x[1])*np.sin(x[0]),-np.sin(x[1])]

def generate_and_save_panoramas(testing=False,dataset_list=["1","2","8","9"]):
        
        for dataset in dataset_list:
            if(not testing or dataset == "10" or dataset == "11"):
                [cam1data,imu,vicon1data] = load_data.load("../../trainset",dataset=dataset,testing=False,camera_data=True)
                cam_ts = np.squeeze(cam1data['ts'],0)
                imu_ts = np.squeeze(vicon1data['ts'],0)
                rots = vicon1data['rots']
            else:
                [cam1data,imu,vicon1data] = load_data.load("../../trainset",dataset=dataset,testing=True,camera_data=True)
                cam_ts = np.squeeze(cam1data['ts'],0)
                imu_ts = np.squeeze(imu['ts'],0)
                qts = np.load("../numpy files" + "/data_set_" + dataset + "_optimized_quarternions.npy")
                rots = np.apply_along_axis(lambda x: transforms3d.quaternions.quat2mat(x),-1,qts).transpose(1,2,0)
            
            sphere_mesh = np.array(np.meshgrid(np.linspace(np.pi/6,-np.pi/6,320),np.linspace(-np.pi/8,np.pi/8,240)))
            sphere_coords = (sphere_mesh).transpose(1,2,0)
            cartesian_coords = np.apply_along_axis(spherical_to_cartesian_coordinates,-1,sphere_coords)

            image = 255*np.ones((960,1280,3),dtype='uint8')

            for i in tqdm(range(cam1data['cam'].shape[-1])):

                img = cam1data['cam'][:,:,:,i]
                rot = rots[...,np.argmin(np.abs(imu_ts-cam_ts[i]))]
                rot_cartesian_coords = np.apply_along_axis(rotate_cartesian_coordinates,-1,cartesian_coords)
                rot_sphere_coords = np.apply_along_axis(rotate_spherical_coordinates,-1,rot_cartesian_coords)
                panorama_coords = np.apply_along_axis(get_image_plane_coordinates,-1,rot_sphere_coords)

                image[panorama_coords[...,0],panorama_coords[...,1]] = img

            fig, axes = plt.subplots(1, 1)
            axes.imshow(image)
            axes.set_title(f'Dataset {dataset}')
            fig.savefig(f"../panoramas/dataset_{dataset}_panorama.jpg")

if __name__=="__main__":
    generate_and_save_panoramas()

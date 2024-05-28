from scipy.signal import resample
from scipy.fft import fft, fftfreq, fftshift
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

FPS = 25

def resample_sequence(sequence, factor=0.9):
    """
    Resamples a sequence to a fixed length.
    """
    num = int(sequence.shape[0]*factor)
    return resample(sequence, num, axis=0)

def plot_pose_sequences(original, resampled, ID):
    """
    Plot the pose sequences and the smoothed pose sequences.
    """
    n_frames, n_joints, dim = original.shape

    fig, axs = plt.subplots(n_joints, 1, figsize=(15, 5*n_joints))
    for joint in range(n_joints):
        axs[joint].plot(original[:, joint, 0], label='x_0')
        axs[joint].plot(original[:, joint, 1], label='y_0')
        axs[joint].plot(resampled[:, joint, 0], label='x_resampled')
        axs[joint].plot(resampled[:, joint, 1], label='y_resampled')
        axs[joint].set_title(f'Joint {joint} - Original')
        # axs[joint, 0].set_xlim(-0.5, x_lim + 0.5)
        axs[joint].legend()

        # axs[joint, 1].plot(resampled[:, joint, 0], label='x')
        # axs[joint, 1].plot(resampled[:, joint, 1], label='y')
        # axs[joint, 1].set_title(f'Joint {joint} - Resampled')
        # axs[joint, 0].set_xlim(-0.5, x_lim + 0.5)
        # axs[joint, 1].legend()

    plt.savefig(f'data/pose_sequence_vs_resampled_pose_sequence_{ID}.png')


def plot_frequency_distribution(original, f_orig, resampled, f_resampled, ID):
    """
    Plot the frequency distribution of the pose sequences.
    """
    n_joints = original.shape[1]
    fig, axs = plt.subplots(n_joints, 2, figsize=(15, 5*n_joints))
    # fig, axs = plt.subplots(2, 1, figsize=(15, 10))
    for joint in range(n_joints):
        axs[joint,0].plot(original[:, joint, 0], np.abs(f_orig), label='x_0')
        axs[joint,0].plot(original[:, joint, 1], np.abs(f_orig), label='y_0')
        axs[joint,0].set_title('Original')
        axs[joint,0].legend()
        axs[joint,1].plot(resampled[:, joint, 0], np.abs(f_resampled), label='x_resampled')
        axs[joint,1].plot(resampled[:, joint, 1], np.abs(f_resampled), label='y_resampled')
        axs[joint,1].set_title('Resampled')
        axs[joint,1].legend()
    
    plt.savefig(f'data/frequency_distribution_{ID}.png')
    
    
     


def get_frequency_distribution(pose_sequence):
    """
    Get the frequency distribution of the pose sequence.
    """
    data_points = pose_sequence.shape[0]
    sequence_f = fft(pose_sequence, axis=0)
    f = fftfreq(data_points, 1/FPS)
    return sequence_f, f
    


def main():
    data_folder = "/Midgard/Data/tibbe/datasets/own/poses_smooth_np/"
    # pose_file = "ID_0011_0.npy"
    for pose_file in os.listdir(data_folder):
        with open(os.path.join(data_folder, pose_file), 'rb') as file:
                data = np.load(file)
        
        pose_sequence = data

        pose_sequence_f, f_orig = get_frequency_distribution(pose_sequence)

        # calculate frequency range where 75% of the energy is
        energy = np.abs(pose_sequence_f[:,0,0])**2
        energy /= energy.sum()
        cum_energy = np.cumsum(energy)
        mask = cum_energy < 0.9
        f_orig = f_orig[mask]

        frequency_range = [min(f_orig), max(f_orig)]

        print(pose_file.split('_')[1])
        print(frequency_range)
        

    # factors = np.linspace(0.5, 2, 16)

    # for i, factor in enumerate(factors):
    #     resampled = resample_sequence(pose_sequence, factor)
    #     resampled_f, f_resampled = get_frequency_distribution(resampled)
    #     # plot_pose_sequences(pose_sequence_f, resampled_f, ID=i)
    #     plot_frequency_distribution(pose_sequence_f, f_orig, resampled_f, f_resampled, ID=i)


def plot_skeletons():
    data_folder = "/Volumes/USB3/thesis/data/poses_smooth_np/"
    out_dir = "/Volumes/USB3/thesis/data/np_poses_corrected/"
    os.makedirs(out_dir, exist_ok=True)
    for pose_file in os.listdir(data_folder):
        with open(os.path.join(data_folder, pose_file), 'rb') as file:
                data = np.load(file)
        
        pose_sequence = data

        pose_sequence = pose_sequence[:,:14,:2]
        n_frames, n_joints, dim = pose_sequence.shape

        # find angle of the body
        hip_point = (pose_sequence[:, 8] + pose_sequence[:, 11]) / 2
        angle = np.arctan2(pose_sequence[:, 1, 1] - hip_point[:, 1], pose_sequence[:, 1, 0] - hip_point[:, 0])
        angle_to_rotate = np.pi/2 - np.mean(angle)
        # rotate the skeleton around the neck joint
        # pose_sequence = np.dot(np.array([[np.cos(angle_to_rotate), -np.sin(angle_to_rotate)], [np.sin(angle_to_rotate), np.cos(angle_to_rotate)]]), pose_sequence.T).T
        for i in range(n_frames):
            pose_sequence[i] = np.dot(np.array([[np.cos(angle_to_rotate), -np.sin(angle_to_rotate)], [np.sin(angle_to_rotate), np.cos(angle_to_rotate)]]), pose_sequence[i].T).T

        hip_point = (pose_sequence[:, 8] + pose_sequence[:, 11]) / 2
        angle = np.arctan2(pose_sequence[:, 1, 1] - hip_point[:, 1], pose_sequence[:, 1, 0] - hip_point[:, 0])


        pose_sequence = np.clip(pose_sequence, -2, 2)
        if np.max(pose_sequence) > 2 or np.min(pose_sequence) < -2:
            print(f"ID: {pose_file.split('_')[1]}")
            print(f"MAX: {np.argmax(pose_sequence, axis=0)}")
            print(f"MIN: {np.argmin(pose_sequence, axis=0)}")

        start = end = int(n_frames*0.05)
        
        pose_sequence = pose_sequence[start:-end]
        n_frames = pose_sequence.shape[0]

        # shift skeleton, so that the hips are at the origin
        hip_point = (pose_sequence[:, 8] + pose_sequence[:, 11]) / 2
        hip_point = np.expand_dims(hip_point, axis=1)
        pose_sequence = pose_sequence - hip_point

        # scale between -1 and 1
        pose_sequence = pose_sequence / np.max(np.abs(pose_sequence))

        fig, axs = plt.subplots(5, 1, figsize=(5, 5*5))

        for j, i in enumerate(np.linspace(0, n_frames-1, 5).astype(int)):
            axs[j].scatter(pose_sequence[i, :, 0], pose_sequence[i, :, 1])
            # draw the edges of the skeleton
            for edge in [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [2, 8], [8, 9], [9, 10], [8,11], [5, 11], [11, 12], [12, 13]]:
                axs[j].plot(pose_sequence[i, edge, 0], pose_sequence[i, edge, 1], color='black')
            
            # define point between hips
            hip_point = (pose_sequence[i, 8] + pose_sequence[i, 11]) / 2


            axs[j].plot([pose_sequence[i, 1, 0], hip_point[0]], [pose_sequence[i, 1, 1], hip_point[1]], color='black')


                # axs[j].set_xlim(-1, 1)
                # axs[j].set_ylim(-1, 1)
            axs[j].set_title(f"Skeleton at frame {i}")
        
        plt.savefig(f"output/sanity_check/skeleton_{pose_file.split('_')[1]}.png")
        plt.close()

        np.save(f"{out_dir}{pose_file}", pose_sequence)


def plot_skeleton_npy():
    data_folder = "/Volumes/USB3/thesis/data/np_poses_corrected/"
    
    for pose_file in os.listdir(data_folder):
        with open(os.path.join(data_folder, pose_file), 'rb') as file:
                data = np.load(file, allow_pickle=True)
        
        pose_sequence = data

        pose_sequence = pose_sequence[:,:14,:2]
        n_frames, n_joints, dim = pose_sequence.shape

        fig, axs = plt.subplots(5, 1, figsize=(5, 5*5))

        for j, i in enumerate(np.linspace(0, n_frames-1, 5).astype(int)):
            axs[j].scatter(pose_sequence[i, :, 0], pose_sequence[i, :, 1])
            # draw the edges of the skeleton
            for edge in [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [2, 8], [8, 9], [9, 10], [8,11], [5, 11], [11, 12], [12, 13]]:
                axs[j].plot(pose_sequence[i, edge, 0], pose_sequence[i, edge, 1], color='black')
            
            # define point between hips
            hip_point = (pose_sequence[i, 8] + pose_sequence[i, 11]) / 2


            axs[j].plot([pose_sequence[i, 1, 0], hip_point[0]], [pose_sequence[i, 1, 1], hip_point[1]], color='black')


                # axs[j].set_xlim(-1, 1)
                # axs[j].set_ylim(-1, 1)
            axs[j].set_title(f"Skeleton at frame {i}")
            
        plt.savefig(f"output/sanity_check/skeleton_{pose_file.split('_')[1]}_npy.pdf")
        plt.close()


def plot_skeleton_pkl():
    data_folder = "/Users/magnusrubentibbe/Dropbox/Magnus_Ruben_TIBBE/Uni/Master_KTH/Thesis/code/data/dataset_KI/poses_renamed/"
    pkl_files = os.listdir(data_folder)
    for pkl_file in pkl_files[:5]:
        with open(os.path.join(data_folder, pkl_file), 'rb') as file:
                data = pd.read_pickle(file)
        
        pose_sequence = get_joint_trajectories_from_df(data)
        # remove the first and last 5% of the sequence
        start = end = int(pose_sequence.shape[0]*0.05)
        pose_sequence = pose_sequence[start:-end]

        pose_sequence = pose_sequence[:,:14,:2]
        n_frames, n_joints, dim = pose_sequence.shape

        fig, axs = plt.subplots(5, 1, figsize=(5, 5*5))

        for j, i in enumerate(np.linspace(0, n_frames-1, 5).astype(int)):
            axs[j].scatter(pose_sequence[i, :, 0], pose_sequence[i, :, 1])
            # draw the edges of the skeleton
            for edge in [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [2, 8], [8, 9], [9, 10], [8,11], [5, 11], [11, 12], [12, 13]]:
                axs[j].plot(pose_sequence[i, edge, 0], pose_sequence[i, edge, 1], color='black')
            
            # define point between hips
            hip_point = (pose_sequence[i, 8] + pose_sequence[i, 11]) / 2


            axs[j].plot([pose_sequence[i, 1, 0], hip_point[0]], [pose_sequence[i, 1, 1], hip_point[1]], color='black')


                # axs[j].set_xlim(-1, 1)
                # axs[j].set_ylim(-1, 1)
            axs[j].set_title(f"Skeleton at frame {i}")
            
        plt.savefig(f"output/sanity_check/skeleton_{pkl_file.split('_')[1]}_pkl.pdf")
        plt.close()


def get_joint_trajectories_from_df(data: pd.DataFrame):

    pose_sequence = (-1)*np.ones((len(data), 18, 3))
    t = 0
    for subset, candidate in zip(data["limbs_subset"].items(), data["limbs_candidate"].items()):
        subset = subset[1]
        candidate = candidate[1]
        for j in range(len(subset)):
            for i in range(18):
                index = int(subset[j][i])
                if index != -1:
                    pose_sequence[t, i] = candidate[index, :3]
        t += 1
    
    return pose_sequence


if __name__ == "__main__":
    plot_skeleton_npy()

from scipy.signal import resample
from scipy.fft import fft, fftfreq, fftshift
import os
import numpy as np
import matplotlib.pyplot as plt

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


from omegaconf import OmegaConf

def get_dataset_mean_and_std():
    data_folder = "/Volumes/USB3/thesis/data/poses_smooth_np/"
    means = []
    stds = []
    for pose_file in os.listdir(data_folder):
        with open(os.path.join(data_folder, pose_file), 'rb') as file:
                data = np.load(file)
        
        pose_sequence = data

        means.append(np.mean(pose_sequence[:,:,:2], axis=0))
        stds.append(np.std(pose_sequence[:,:,:2], axis=0))
        
    means = np.array(means)
    stds = np.array(stds)

    mean = np.mean(means, axis=0)
    std = np.mean(stds, axis=0)

    np.save('data/means.npy', mean)
    np.save('data/stds.npy', std)


def get_dataset_max_min():
    data_folder = "/Volumes/USB3/thesis/data/poses_smooth_np/"
    maxs = []
    mins = []
    for pose_file in os.listdir(data_folder):
        with open(os.path.join(data_folder, pose_file), 'rb') as file:
                data = np.load(file)
        
        pose_sequence = data

        maxs.append(np.max(pose_sequence[:,:,:2], axis=0))
        mins.append(np.min(pose_sequence[:,:,:2], axis=0))

    maxs = np.array(maxs)
    mins = np.array(mins)

    max = np.mean(maxs, axis=0)
    min = np.mean(mins, axis=0)

    np.save('data/maxs.npy', max)
    np.save('data/mins.npy', min)



    # factors = np.linspace(0.5, 2, 16)

    # for i, factor in enumerate(factors):
    #     resampled = resample_sequence(pose_sequence, factor)
    #     resampled_f, f_resampled = get_frequency_distribution(resampled)
    #     # plot_pose_sequences(pose_sequence_f, resampled_f, ID=i)
    #     plot_frequency_distribution(pose_sequence_f, f_orig, resampled_f, f_resampled, ID=i)


if __name__ == "__main__":
    get_dataset_max_min()

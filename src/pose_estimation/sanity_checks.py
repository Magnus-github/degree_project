import matplotlib.pyplot as plt
import numpy as np
import pickle
import os


def plot_pose_sequences(pose_sequence, smoothed_pose_sequence, out_folder, ID):
    """
    Plot the pose sequences and the smoothed pose sequences.

    In:
        pose_sequence: np.array, shape=(n_frames, n_joints, 3)
        smoothed_pose_sequence: np.array, shape=(n_frames, n_joints, 3)
    """
    n_frames, n_joints, dim = pose_sequence.shape

    fig, axs = plt.subplots(n_joints, 2, figsize=(15, 5*n_joints))
    for joint in range(n_joints):
        axs[joint, 0].plot(pose_sequence[:, joint, 0], label='x')
        axs[joint, 0].plot(pose_sequence[:, joint, 1], label='y')
        axs[joint, 0].set_title(f'Joint {joint} - Original')
        axs[joint, 0].legend()

        axs[joint, 1].plot(smoothed_pose_sequence[:, joint, 0], label='x')
        axs[joint, 1].plot(smoothed_pose_sequence[:, joint, 1], label='y')
        axs[joint, 1].set_title(f'Joint {joint} - Smoothed')
        axs[joint, 1].legend()

    plt.savefig(out_folder + 'pose_sequence_vs_smoothed_pose_sequence_{}.png'.format(ID))


def main():
    pose_directory = "/Midgard/Data/tibbe/datasets/own/pose_sequences_openpose_renamed_smooth/"
    out_folder = "/Midgard/home/tibbe/thesis/degree_project/output/pose_estimation/sanity_checks/"
    for file in os.listdir(pose_directory):
        if file.endswith(".pkl"):
            df_path = os.path.join(pose_directory, file)
            df_path = pose_directory + "ID_0100_0.pkl"

            with open(df_path, 'rb') as f:
                data = pickle.load(f)

            pose_sequence = np.zeros((len(data['joint_trajectories']), 18, 3))
            smoothed_pose_sequence = np.zeros((len(data['joint_trajectories']), 18, 3))

            i=0
            for pose, smooth_pose in zip(data["joint_trajectories"].items(), data["smoothed_joint_trajectories"].items()):
                frame = i
                pose_sequence[frame] = np.array(pose[1])
                smoothed_pose_sequence[frame] = np.array(smooth_pose[1])
                i+=1

            ID = '_'.join(file.split('_')[1:3])
            ID = '0100_0'
            plot_pose_sequences(pose_sequence, smoothed_pose_sequence, out_folder, ID)

        break
        


if __name__ == "__main__":
    main()
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os


def plot_pose_sequences(pose_sequence, smoothed_pose_sequence, rotated_pose_sequence, normalized_pose_sequence, out_folder, ID):
    """
    Plot the pose sequences and the smoothed pose sequences.

    In:
        pose_sequence: np.array, shape=(n_frames, n_joints, 3)
        smoothed_pose_sequence: np.array, shape=(n_frames, n_joints, 3)
    """
    n_frames, n_joints, dim = pose_sequence.shape

    fig, axs = plt.subplots(n_joints, 4, figsize=(15, 5*n_joints))
    for joint in range(n_joints):
        axs[joint, 0].plot(pose_sequence[:, joint, 0], label='x')
        axs[joint, 0].plot(pose_sequence[:, joint, 1], label='y')
        axs[joint, 0].set_title(f'Joint {joint} - Original')
        axs[joint, 0].legend()

        axs[joint, 1].plot(smoothed_pose_sequence[:, joint, 0], label='x')
        axs[joint, 1].plot(smoothed_pose_sequence[:, joint, 1], label='y')
        axs[joint, 1].set_title(f'Joint {joint} - Smoothed')
        axs[joint, 1].legend()

        axs[joint, 2].plot(rotated_pose_sequence[:, joint, 0], label='x')
        axs[joint, 2].plot(rotated_pose_sequence[:, joint, 1], label='y')
        axs[joint, 2].set_title(f'Joint {joint} - Rotated')
        axs[joint, 2].legend()

        axs[joint, 3].plot(normalized_pose_sequence[:, joint, 0], label='x')
        axs[joint, 3].plot(normalized_pose_sequence[:, joint, 1], label='y')
        axs[joint, 3].set_title(f'Joint {joint} - Normalized')
        axs[joint, 3].legend()

    plt.savefig(out_folder + 'pose_sequence_vs_smoothed_pose_sequence_{}.png'.format(ID))


def main():
    pose_directory = "/Users/magnusrubentibbe/Dropbox/Magnus_Ruben_TIBBE/Uni/Master_KTH/Thesis/code/data/dataset_KI/poses_smooth_debug/"
    out_folder = "data/outputs/"
    for file in os.listdir(pose_directory)[:10]:
        if file.endswith(".pkl"):
            df_path = os.path.join(pose_directory, file)

            with open(df_path, 'rb') as f:
                data = pickle.load(f)

            pose_sequence = np.zeros((len(data['joint_trajectories']), 18, 3))
            smoothed_pose_sequence = np.zeros((len(data['joint_trajectories']), 18, 3))
            rotated_pose_sequence = np.zeros((len(data['joint_trajectories']), 18, 3))
            normalized_pose_sequence = np.zeros((len(data['joint_trajectories']), 18, 3))

            i=0
            for pose, smooth_pose, rotated_pose, normalized_pose in zip(data["joint_trajectories"].items(), data["smoothed_joint_trajectories"].items(), data["rotated_joint_trajectories"].items(), data["normalized_joint_trajectories"].items()):
                frame = i
                pose_sequence[frame] = np.array(pose[1])
                smoothed_pose_sequence[frame] = np.array(smooth_pose[1])
                rotated_pose_sequence[frame] = np.array(rotated_pose[1])
                normalized_pose_sequence[frame] = np.array(normalized_pose[1])
                i+=1

            ID = '_'.join(file.split('_')[1:3])
            plot_pose_sequences(pose_sequence, smoothed_pose_sequence, rotated_pose_sequence, normalized_pose_sequence, out_folder, ID)


if __name__ == "__main__":
    main()

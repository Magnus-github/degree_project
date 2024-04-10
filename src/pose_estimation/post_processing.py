import numpy as np
from scipy.signal import savgol_filter
import json
import argparse
from scipy.interpolate import CubicSpline
import os
import pandas as pd
import pickle
from omegaconf import OmegaConf


def get_joint_trajectories(pose_dict: dict):
    """
    Get joint trajectories from pose_dict.

    :param pose_dict: dict
    pose_dict: {
      "pose_sequence": [{
          "frame_id": id,
          "people": [
                  {  
                      "person_id": id
                      "pose_keypoints_2d": [x1, y1, c1, x2, y2, c2, ...]
                  }
                  ]
         }]
    }
    :return: dict
    joint_trajectories: {
        "person_id": {
            "x": [x1, x2, ...],
            "y": [y1, y2, ...],
            "c": [c1, c2, ...]
        }
    }
    where x1, y1, c1 are the x, y, c of all joints in the first frame.
    """
    sequence = pose_dict['pose_sequence']
    joint_trajectories = {'num_frames': len(sequence)}
    trajectories = {}
    for frame in sequence:
        frame_id = frame['frame_id']
        people = frame['people']
        for person in people:
            person_id = person['person_id']
            pose_keypoints_2d = person['pose_keypoints_2d']
            x = pose_keypoints_2d[0::3]
            y = pose_keypoints_2d[1::3]
            c = pose_keypoints_2d[2::3]
            if person_id not in trajectories:
                trajectories[person_id] = {
                    "x": [],
                    "y": [],
                    "c": []
                }
            trajectories[person_id]['x'].append(x)
            trajectories[person_id]['y'].append(y)
            trajectories[person_id]['c'].append(c)

    for person_id in trajectories:
        trajectories[person_id]['x'] = np.array(trajectories[person_id]['x'])
        trajectories[person_id]['y'] = np.array(trajectories[person_id]['y'])
        trajectories[person_id]['c'] = np.array(trajectories[person_id]['c'])

    joint_trajectories['trajectories'] = trajectories

    return joint_trajectories

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


def smooth_joint_trajectories(joint_trajectories: dict):
    """
    Smooth joint trajectories using Savitzky-Golay filter.

    :param joint_trajectories: dict
    joint_trajectories: {
        "person_id": {
            "x": [x1, x2, ...],
            "y": [y1, y2, ...],
            "c": [c1, c2, ...]
        }
    }
    :return: dict
    smoothed_joint_trajectories: {
        "person_id": {
            "x": [x1, x2, ...],
            "y": [y1, y2, ...],
            "c": [c1, c2, ...]
        }
    }
    """
    num_frames = joint_trajectories['num_frames']
    smoothed_joint_trajectories = {'x': np.zeros((num_frames,18)), 'y': np.zeros((num_frames,18)), 'c': np.zeros((num_frames,18))}
    for person_id in joint_trajectories['trajectories']:
        # only save joint trajectoreis of person if they are present in all frames
        if joint_trajectories['trajectories'][person_id]['x'].shape[0] == joint_trajectories['num_frames']:
            for joint in range(18):
                allX = joint_trajectories['trajectories'][person_id]['x'][:, joint]
                allY = joint_trajectories['trajectories'][person_id]['y'][:, joint]
                allC = joint_trajectories['trajectories'][person_id]['c'][:, joint]

                mean_confidence = np.mean(allC, axis=0)

                smooth_idx = np.where(allC > mean_confidence - 0.1)[0]

                # if smooth_idx[0] > 10:
                #     smooth_idx = np.concatenate((np.array([0]), smooth_idx))

                int_x = CubicSpline(smooth_idx, allX[smooth_idx], bc_type='natural')(range(len(allX)))
                smoothed_x = savgol_filter(int_x, 5, 3)

                int_y = CubicSpline(smooth_idx, allY[smooth_idx], bc_type='natural')(range(len(allY)))
                smoothed_y = savgol_filter(int_y, 5, 3)

                int_c = CubicSpline(smooth_idx, allC[smooth_idx], bc_type='natural')(range(len(allC)))
                smoothed_c = savgol_filter(int_c, 5, 3)

                smoothed_joint_trajectories['x'][:, joint] = smoothed_x
                smoothed_joint_trajectories['y'][:, joint] = smoothed_y
                smoothed_joint_trajectories['c'][:, joint] = int_c

    return smoothed_joint_trajectories


def smooth_joint_trajectories_from_array(joint_trajectories: np.array):
    num_frames = joint_trajectories.shape[0]
    smoothed_joint_trajectories = np.zeros_like(joint_trajectories)

    for joint in range(18):
        mean_confidence = np.mean(joint_trajectories[:, joint, 2], axis=0)

        smooth_idx = np.where(joint_trajectories[:, joint, 2] > mean_confidence - 0.1)[0]
        if smooth_idx[0] > 10:
            smooth_idx = np.concatenate((np.array([0]), smooth_idx))
        if smooth_idx[-1] < num_frames - 10:
            smooth_idx = np.append(smooth_idx, num_frames - 1)

        # # if there is a gap somewhere, add an index in the middle of the gap
        # for i in range(len(smooth_idx) - 1):
        #     if smooth_idx[i+1] - smooth_idx[i] > 20:
        #         smooth_idx = np.append(smooth_idx, (smooth_idx[i+1] + smooth_idx[i]) // 2)


        int_x = CubicSpline(smooth_idx, joint_trajectories[smooth_idx, joint, 0], bc_type='natural')(range(len(joint_trajectories)))
        smoothed_x = savgol_filter(int_x, 5, 3)

        int_y = CubicSpline(smooth_idx, joint_trajectories[smooth_idx, joint, 1], bc_type='natural')(range(len(joint_trajectories)))
        smoothed_y = savgol_filter(int_y, 5, 3)

        int_c = CubicSpline(smooth_idx, joint_trajectories[smooth_idx, joint, 2], bc_type='natural')(range(len(joint_trajectories)))

        smoothed_joint_trajectories[:, joint, 0] = smoothed_x
        smoothed_joint_trajectories[:, joint, 1] = smoothed_y
        smoothed_joint_trajectories[:, joint, 2] = int_c

    return smoothed_joint_trajectories


def smoothing_from_dict(cfg: dict):
    # load pose dict
    pose__estim_out_folder = cfg.POSE_ESTIMATION.OUTPUT_FOLDER

    for pose_file in os.listdir(pose__estim_out_folder):
        if pose_file.endswith('.json'):
            pose_file_path = os.path.join(pose__estim_out_folder, pose_file)
            with open(pose_file_path) as f:
                pose_dict = json.load(f)

            # get joint trajectories
            joint_trajectories = get_joint_trajectories(pose_dict)

            # smooth joint trajectories
            smoothed_joint_trajectories = smooth_joint_trajectories(joint_trajectories)

            # save smoothed joint trajectories
            out_path = os.path.join(cfg.POSE_ESTIMATION.OUTPUT_FOLDER, '_'.join([pose_file.split('.')[0], 'smoothed.json']))
            with open(out_path, 'w') as f:
                json.dump(smoothed_joint_trajectories, f)

            print(f"Saved smoothed joint trajectories to {out_path}")


def rotate_relative(joints, center, angle):
    x = joints[:, :, 0] - center[0]
    y = joints[:, :, 1] - center[1]
    joints[:, :, 0] = x * np.cos(angle) - y * np.sin(angle) + center[0]
    joints[:, :, 1] = x * np.sin(angle) + y * np.cos(angle) + center[1]
    return joints

def rotate_joint_trajectories(joint_trajectories: np.array):
    # calculate angle of shoulders
    ang_shoulder = np.arctan2(joint_trajectories[:, 5, 1] - joint_trajectories[:, 2, 1], joint_trajectories[:, 5, 0] - joint_trajectories[:, 2, 0])
    ang_shoulder = np.mean(ang_shoulder)
    # calculate angle of hips
    ang_hips = np.arctan2(joint_trajectories[:, 11, 1] - joint_trajectories[:, 8, 1], joint_trajectories[:, 11, 0] - joint_trajectories[:, 8, 0])
    ang_hips = np.mean(ang_hips)
    # calculate center of body
    center_shoulder = (joint_trajectories[:, 5, :2] + joint_trajectories[:, 2, :2]) / 2
    center_shoulder = np.mean(center_shoulder, axis=0)
    center_hips = (joint_trajectories[:, 11, :2] + joint_trajectories[:, 8, :2]) / 2
    center_hips = np.mean(center_hips, axis=0)
    center_body = (center_shoulder + center_hips) / 2


    # rotate joints around center of body
    # left arm:
    joint_trajectories[:, 2:5, :2] = rotate_relative(joint_trajectories[:, 2:5, :2], center_body, ang_shoulder)
    # right arm:
    joint_trajectories[:, 5:8, :2] = rotate_relative(joint_trajectories[:, 5:8, :2], center_body, ang_shoulder)
    # head:
    joint_trajectories[:, 0:2, :2] = rotate_relative(joint_trajectories[:, 0:2, :2], center_body, ang_shoulder)
    joint_trajectories[:, 14:18, :2] = rotate_relative(joint_trajectories[:, 14:18, :2], center_body, ang_shoulder)
    # right leg:
    joint_trajectories[:, 8:11, :2] = rotate_relative(joint_trajectories[:, 8:11, :2], center_body, ang_hips)
    # left leg:
    joint_trajectories[:, 11:14, :2] = rotate_relative(joint_trajectories[:, 11:14, :2], center_body, ang_hips)

    return joint_trajectories
    
def normalize_joint_trajectories(joint_trajectories: np.array):
    ref_joint = joint_trajectories[:, 1, :2]
    ref_joint = np.mean(ref_joint, axis=0)
    center_hips = (joint_trajectories[:, 11, :2] + joint_trajectories[:, 8, :2]) / 2
    center_hips = np.mean(center_hips, axis=0)
    ref_dist = np.linalg.norm(ref_joint - center_hips)

    for i in range(18):
        joint_trajectories[:, i, :2] = joint_trajectories[:, i, :2] - ref_joint
        joint_trajectories[:, i, :2] = joint_trajectories[:, i, :2] / ref_dist

    return joint_trajectories

def smoothing_from_df(cfg: dict):
    # load pose dict
    pose__estim_out_folder = cfg.POSES.PATH

    for i, pose_file in enumerate(os.listdir(pose__estim_out_folder)):
        if pose_file.endswith('.pkl'):
            pose_file_path = os.path.join(pose__estim_out_folder, pose_file)
            with open(pose_file_path, 'rb') as f:
                data = pd.read_pickle(f)

            # get joint trajectories
            joint_trajectories = get_joint_trajectories_from_df(data)
            data["joint_trajectories"] = joint_trajectories.tolist()

            # smooth joint trajectories
            smoothed_joint_trajectories = smooth_joint_trajectories_from_array(joint_trajectories)
            if cfg.POST_PROCESS.DEBUG:
                data["smoothed_joint_trajectories"] = smoothed_joint_trajectories.tolist()

            # rotate joint trajectories
            smoothed_joint_trajectories  = rotate_joint_trajectories(smoothed_joint_trajectories)
            if cfg.POST_PROCESS.DEBUG:
                data["rotated_joint_trajectories"] = smoothed_joint_trajectories.tolist()

            # normalize joint trajectories
            smoothed_joint_trajectories = normalize_joint_trajectories(smoothed_joint_trajectories)
            if cfg.POST_PROCESS.DEBUG:
                data["normalized_joint_trajectories"] = smoothed_joint_trajectories.tolist()
            
            if not cfg.POST_PROCESS.DEBUG:
                data["smoothed_joint_trajectories"] = smoothed_joint_trajectories.tolist()

                out_np = os.path.join(cfg.POSES.SMOOTH_OUT_PATH, pose_file.split('.')[0] + '.npy')
                np.save(out_np, smoothed_joint_trajectories)

            # save updated dataframe
            out_path = os.path.join(cfg.POSES.SMOOTH_OUT_PATH, pose_file)
            with open(out_path, 'wb') as f:
                pickle.dump(data, f)

            print(f"Saved smoothed joint trajectories to {out_path}")
            print(i)
            

def main(cfg: dict):

    mode = cfg.POST_PROCESS.MODE
    if mode == "from_dict":
        smoothing_from_dict(cfg)
    elif mode == "from_df":
        smoothing_from_df(cfg)
    else:
        print("Invalid mode. Use 'from_dict' or 'from_df'.")    
    

    # # display joint trajectories
    # test_video = 'ID01_fullterm_hypotermi_HINE21_MR djup asfyxi_13w F- (Nemo)_anon.mp4'
    # cap = cv2.VideoCapture(test_video)
    # frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # fps = int(cap.get(cv2.CAP_PROP_FPS))
    # num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # num_ann_frames = joint_trajectories['num_frames']
    # # out = cv2.VideoWriter('out.avi',cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width,frame_height))
    # out_smooth = cv2.VideoWriter('out_smooth.avi',cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width,frame_height))

    # try:
    #     print("Drawing joint trajectories on video...")
    #     frame_id = 0
    #     # while cap.isOpened():
    #     for _ in tqdm(range(num_ann_frames)):
    #         if not cap.isOpened():
    #             print("Capture closed.")
    #             break
    #         ret, frame = cap.read()
    #         if frame is None or frame_id > joint_trajectories['num_frames']:
    #             break
    #         jTrajectory = joint_trajectories['trajectories'][0]
    #         # canvas = frame.copy()
    #         # frame_joints = util.draw_bodypose_from_jointTrajectory(canvas, jTrajectory, frame_id)
    #         canvas = frame.copy()
    #         frame_smoothjoints = util.draw_bodypose_from_jointTrajectory(canvas, smoothed_joint_trajectories, frame_id)

    #         frame_id += 1

    #         # out.write(frame_joints)
    #         out_smooth.write(frame_smoothjoints)

    # except KeyboardInterrupt:
    #     print("Interrupted, cleaning up...")

    # finally:
    #     print("Cleaning up...")
    #     cap.release()
    #     # out.release()
    #     out_smooth.release()
    #     print("Done.")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Post-process pose estimation output")
    parser.add_argument("--config", type=str, default="config/pose_extraction_config.yaml", help="Path to the config file")
    args = parser.parse_args()


    cfg = OmegaConf.load(args.config)
    main(cfg)


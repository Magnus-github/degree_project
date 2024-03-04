from body import Body
import util

import keras
import torch
import cv2
import numpy as np

import argparse
import time
from omegaconf import OmegaConf
import os
import json



def load_keras_weights(body_estimation_model: Body, keras_weights: str):
    model = keras.models.load_model(keras_weights)

    # print(model.summary())

    # load the weights of model into body_estimation
    filled = False
    for i, layer in enumerate(model.layers):
        if i < 2:
            print(layer.name)
            continue
        weights = layer.get_weights()
        if len(weights) == 0:
            continue
        print(f"Layer {i-2} ({layer.name}): {weights[0].shape}, {weights[1].shape}")
        for d_name, d_layer in body_estimation_model.model.named_children():
            if filled:
                break
            if 'model' in d_name:
                for d_name_, d_layer_ in d_layer.named_children():
                    if layer.name in d_name_ and not filled:
                        print(f"Setting weights for {d_name_}")
                        d_layer_.weight.data = torch.tensor(weights[0].transpose(3, 2, 0, 1))
                        d_layer_.bias.data = torch.tensor(weights[1])
                        filled = True
                        break
            else:
                if layer.name in d_name and not filled:
                    print(f"Setting weights for {d_name}")
                    d_layer.weight.data = torch.tensor(weights[0].transpose(3, 2, 0, 1))
                    d_layer.bias.data = torch.tensor(weights[1])
                    filled = True
                    break
        filled = False

def estimate_pose(video_path: str, body_estimation: Body):
    cap = cv2.VideoCapture(video_path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video length: {num_frames} frames")
    sequence = []
    n = 0
    try:
        for i in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                break
            start = time.time()
            candidate, subset = body_estimation(frame)
            print("Processing time: {:.4f} sec".format(time.time() - start))
            # canvas = util.draw_bodypose(frame, candidate, subset)
            # cv2.imwrite('test_data/result.png', canvas)
            keypoints = (-1)*np.ones((18, 3))
            people = []
            for j in range(len(subset)):
                for i in range(18):
                    index = int(subset[j][i])
                    if index != -1:
                        keypoints[i] = candidate[index, :3]
                people.append({"person_id": j, "pose_keypoints_2d": keypoints.flatten().tolist()})
            
            sequence.append({"frame_id": n, "people": people})

            n += 1

    except Exception as e:
        print(f"Error: {e}")

    finally:
        print("Cleaning up and writing output...")
        cap.release()
        out_dict = {
            "body_pose": "coco",
            "body_parts": ["Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", "LWrist", "RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle", "REye", "LEye", "REar", "LEar"],
            "pairs": [[1,2], [1,5], [2,3], [3,4], [5,6], [6,7], [1,8], [8,9], [9,10], [1,11], [11,12], [12,13], [1,0], [0,14], [14,16], [0,15], [15,17], [2,16], [5,17]],
            "pose_sequence": sequence
        }

        return out_dict


def main(cfg: dict):
    body_estimation = Body(cfg.MODEL.PATH) 
    load_finetuned_weights = cfg.MODEL.LOAD_FINETUNED_WEIGHTS
    video_folder = cfg.POSE_ESTIMATION.VIDEO_FOLDER

    if load_finetuned_weights:
        keras_weights = cfg.MODEL.FINETUNED_WEIGHTS_PATH
        load_keras_weights(body_estimation, keras_weights)

    try:
        for video in os.listdir(video_folder):
        # video = os.listdir(video_folder)[0]
            video_path = os.path.join(video_folder, video)
            print(f"Processing {video_path}")
            out_dict = estimate_pose(video_path, body_estimation)
            
            out_path = os.path.join(cfg.POSE_ESTIMATION.OUTPUT_FOLDER, '_'.join([video.split('.')[0], 'pose_keypoints.json']))
            with open(out_path, 'w') as f:
                json.dump(out_dict, f)

            if KeyboardInterrupt:
                break
            
    except KeyboardInterrupt:
        print("Interrupted, cleaning up...")

    finally:
        print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pose estimation")
    parser.add_argument("--cfg_path", type=str, default="degree_project/src/config/pose_extraction_config.yaml", help="Path to config file")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.cfg_path)
    main(cfg)
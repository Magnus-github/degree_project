import torch
from torch.utils.data import Dataset
import csv
import pickle
import pandas as pd
import numpy as np
import os
import time
import random


class KIDataset(Dataset):
    def __init__(self, data_folder: str, annotations_path: str, mode: str = "train", seed: int = 42):
        self.data_folder = data_folder
        random.seed(seed)
        train_data = random.sample(os.listdir(self.data_folder), int(0.8*len(os.listdir(self.data_folder))))
        val_test_data = list(set(os.listdir(self.data_folder)) - set(train_data))
        val_data = random.sample(val_test_data, int(0.5*len(val_test_data)))
        test_data = list(set(val_test_data) - set(val_data))
        if mode == "train":
            self.data = train_data
        if mode == "val":
            self.data = val_data
        if mode == "test":
            self.data = test_data

        self.labels = []
        self.ids = {}
        self._get_labels_and_ids(annotations_path)


    def __len__(self):
        return len(self.data)
    
    def _get_labels_and_ids(self, annotations_path: str):
        with open(annotations_path, 'r', encoding='utf-8-sig') as file:
            reader = csv.reader(file, delimiter=';', )
            for i,row in enumerate(reader):
                if i == 0:
                    continue
                self.ids[int(row[0])] = i-1
                self.labels.append(row[1])

    
    def __getitem__(self, idx):
        pose_file = self.data[idx]
        with open(os.path.join(self.data_folder, pose_file), 'rb') as file:
            data = pickle.load(file)

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

        id = int(pose_file.split("_")[1])
        label = self.labels[self.ids[id]]
        return pose_sequence, label


if __name__ == "__main__":
    data_folder = "/Midgard/Data/tibbe/datasets/own/pose_sequences_openpose_renamed/"
    annotations_path = "/Midgard/Data/tibbe/datasets/own/annotations.csv"
    d = KIDataset(data_folder=data_folder, annotations_path=annotations_path)

    dataloader = torch.utils.data.DataLoader(d, batch_size=1, shuffle=True)

    print(len(d))
    start = time.time()
    t = start
    label_lst  = []
    seq_lens = []
    for i, (pose_sequence, label) in enumerate(dataloader):
        B, T, J, C = pose_sequence.shape
        label_lst.append(int(label[0]))
        seq_lens.append(T)

        # Reshape pose_sequence to (B, T, J, 1, C)
        # pose_sequence_reshaped = pose_sequence.unsqueeze(3)

        # # Compute absolute differences for both x and y dimensions
        # abs_diff = torch.abs(pose_sequence_reshaped - pose_sequence_reshaped.permute(0, 1, 3, 2, 4))

        # # Extract x and y distances
        # distance_x = abs_diff[:, :, :, :, 0].unsqueeze(4)
        # distance_y = abs_diff[:, :, :, :, 1].unsqueeze(4)

        # # Concatenate distance tensors along dimension 3
        # features = torch.cat((distance_x, distance_y), dim=4)
        # print(pose_sequence.shape, label)
        # print(f"Time: {time.time() - t}")
        # t = time.time()

    print(f"Total time: {time.time() - start}")
    print(f"Average sequence length: {np.mean(seq_lens)}")
    print(f"Max sequence length: {np.max(seq_lens)}")
    print(f"Min sequence length: {np.min(seq_lens)}")
    print(f"Label distribution: {np.bincount(label_lst)}")


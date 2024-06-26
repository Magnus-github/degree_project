import torch
from torch.utils.data import Dataset
import csv
import numpy as np
import os
import time
from tqdm import tqdm
import random
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold

from typing import Optional


class KIDataset(Dataset):
    def __init__(self, data_folder: str, annotations_path: str, num_folds: int = 10,mode: str = "train", transform: Optional[object] = None, seed: int = 42, test_fold: int = 0, val_fold: int = 0):
        self.labels = []
        self.ids = {}
        labels = self._get_labels_and_ids(annotations_path)

        self.data_folder = data_folder
        all_data = os.listdir(self.data_folder)
        all_labels = [labels[self.ids[int(file.split("_")[1])]] for file in all_data]

        # Split data into train/val and test
        skf = StratifiedKFold(n_splits=num_folds, random_state=seed, shuffle=True)
        splits = list(skf.split(all_data, all_labels))
        train_val_idx, test_idx = splits[test_fold]

        # Split train/val into train and val
        skf_2 = StratifiedKFold(n_splits=num_folds-1, random_state=seed, shuffle=True)
        train_val_data = [all_data[i] for i in train_val_idx]
        train_val_labels = [all_labels[i] for i in train_val_idx]
        splits_2 = list(skf_2.split(train_val_data, train_val_labels))
        train_idx, val_idx = splits_2[val_fold]

        if mode == "train":
            self.data = [train_val_data[i] for i in train_idx]
            self.labels = [train_val_labels[i] for i in train_idx]
        if mode == "val":
            self.data = [train_val_data[i] for i in val_idx]
            self.labels = [train_val_labels[i] for i in val_idx]
        if mode == "test":
            self.data = [all_data[i] for i in test_idx]
            self.labels = [all_labels[i] for i in test_idx]
        if mode == "all":
            self.data = all_data
            self.labels = all_labels

        self.transform = transform


    def __len__(self):
        return len(self.data)
    
    def _get_labels_and_ids(self, annotations_path: str):
        labels = []
        with open(annotations_path, 'r', encoding='utf-8-sig') as file:
            reader = csv.reader(file, delimiter=';', )
            for i,row in enumerate(reader):
                if i == 0:
                    continue
                self.ids[int(row[0])] = i-1
                labels.append(int(row[1]))

        return labels

    
    def __getitem__(self, idx):
        pose_file = self.data[idx]
        with open(os.path.join(self.data_folder, pose_file), 'rb') as file:
            data = np.load(file)
        
        pose_sequence = data

        label = self.labels[idx]

        if self.transform and self.transform.class_agnostic:
            pose_sequence = self.transform(pose_sequence)
        elif self.transform and not self.transform.class_agnostic:
            if label == 1 or label == 4:
                pose_sequence = self.transform(pose_sequence)

        return pose_sequence, label
    

class KIDataset_dynamicClipSample(KIDataset):
    def __init__(self, data_folder: str="/Midgard/Data/tibbe/datasets/own/poses_smooth_np/",
                 annotations_path: str="/Midgard/Data/tibbe/datasets/own/annotations.csv",
                 num_folds: int = 10, test_fold: int = 0, val_fold: int = 0,
                 mode: str = "train", transform = None, seed: int = 42,
                 sample_rate: int = 2, clip_length: int = 720, max_overlap: int = 50):
        super().__init__(data_folder, annotations_path, num_folds, mode, transform, seed, test_fold, val_fold)
        self.sample_rate = sample_rate
        self.clip_length = clip_length
        self.stride = clip_length - max_overlap

    def __getitem__(self, idx):
        pose_file = self.data[idx]
        with open(os.path.join(self.data_folder, pose_file), 'rb') as file:
            data = np.load(file)
        
        pose_sequence = data

        label = self.labels[idx]

        if self.transform and self.transform.class_agnostic:
            pose_sequence = self.transform(pose_sequence)
        elif self.transform and not self.transform.class_agnostic:
            if label == 1 or label == 4:
                pose_sequence = self.transform(pose_sequence)

        n_frames = pose_sequence.shape[0]

        # pose_clips = []

        # if (n_frames - self.clip_length) / self.stride < 1: # If we can't take at least one stride (i.e. two clips) from the sequence
        #     self.sample_rate = 1

        # if n_frames < self.clip_length: # If the sequence is shorter than the clip length
        #     pose_sequence = np.concatenate([pose_sequence, np.zeros((self.clip_length - n_frames, pose_sequence.shape[1], pose_sequence.shape[2]))], axis=0)
        #     pose_clips.append(pose_sequence)
        # elif n_frames > self.clip_length:
        #     sample_pool = list(range(0, n_frames - self.clip_length, self.stride))
        #     for i in range(self.sample_rate):
        #         start = random.choice(sample_pool)
        #         pose_clips.append(pose_sequence[start:start+self.clip_length])
        #         sample_pool.remove(start)

        # pose_clips = np.array(pose_clips)
        
        assert n_frames >= self.clip_length, f"Clip length {self.clip_length} is longer than sequence length {n_frames}."
        pose_clips = torch.tensor(pose_sequence).unfold(0, self.clip_length, self.stride).permute(0, 3, 1, 2).contiguous()

        label = torch.tensor([label]*pose_clips.shape[0])

        return pose_clips, label


def collate_fn(batch):
    pose_sequences = [item[0] for item in batch]

    pose_sequences = torch.concat(pose_sequences, dim=0)
    labels = [item[1] for item in batch]
    labels = torch.tensor(labels)
    return pose_sequences, labels

if __name__ == "__main__":
    data_folder = "/Midgard/Data/tibbe/datasets/own/poses_smooth_np/"
    # data_folder = "/Users/magnusrubentibbe/Dropbox/Magnus_Ruben_TIBBE/Uni/Master_KTH/Thesis/code/data/dataset_KI/poses_smooth_np/"
    annotations_path = "/Midgard/Data/tibbe/datasets/own/annotations.csv"
    # annotations_path = "/Users/magnusrubentibbe/Dropbox/Magnus_Ruben_TIBBE/Uni/Master_KTH/Thesis/code/data/dataset_KI/annotations.csv"
    d_train = KIDataset(data_folder=data_folder, annotations_path=annotations_path, mode="train")
    d_val = KIDataset(data_folder=data_folder, annotations_path=annotations_path, mode="val")



    weights = [2.61538462, 1.61904762, 0.5]
    mapping = {'1': 0, '4': 1, '12': 2}
    labels_train = [d_train.labels[d_train.ids[int(file.split("_")[1])]] for file in d_train.data]
    y_train = [mapping[label] for label in labels_train]
    class_sample_count = np.array([len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in y_train])
    samples_weight = torch.from_numpy(samples_weight)
    num_samples = int(max(class_sample_count)*len(class_sample_count))
    # num_samples = 6
    sampler = torch.utils.data.WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), num_samples)
    # sampler = torch.utils.data.WeightedRandomSampler(weights, len(d_train), replacement=True)
    dataloader = torch.utils.data.DataLoader(d_train, batch_size=1)
    start = time.time()
    t = start
    label_lst  = []
    seq_lens = []
    for i, (pose_sequence, label) in enumerate(tqdm(dataloader)):
        B, T, J, C = pose_sequence.shape
        label_lst.append(mapping[label[0]])
        seq_lens.append(T)

        print(pose_sequence.shape, label)

        # fig, axs = plt.subplots(J, 1, figsize=(15, 5*J))
        # for joint in range(J):
        #     axs[joint].plot(pose_sequence[0, :, joint, 0], label='x_0')
        #     axs[joint].plot(pose_sequence[0, :, joint, 1], label='y_0')
        #     axs[joint].set_title(f'Joint {joint} - Original')
        #     axs[joint].legend()
        
        # plt.savefig(f'data/pose_augmented{i}.png')

        if i == 10:
            break

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

    print(i)

    print(f"Total time: {time.time() - start}")
    print(f"Average sequence length: {np.mean(seq_lens)}")
    print(f"Max sequence length: {np.max(seq_lens)}")
    print(f"Min sequence length: {np.min(seq_lens)}")
    print(f"Label distribution: {np.bincount(label_lst)}")

    classes = np.unique(label_lst)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=label_lst)
    print(f"Class weights: {class_weights}")


import os
import torch
from torch.utils.data import Dataset
import torchvision
import glob
import csv
import numpy as np
import matplotlib.pyplot as plt

class UCFDataset(Dataset):
    def __init__(self,
                 split=None,
                 data_root="data/ucf_hmdb/ucf/video",
                 video_extensions=(".mp4", ".avi"),
                 ):
        self.data_root = data_root
        self.split = split
        self.video_extensions = video_extensions

        # Placeholder (not used)
        self.actions = [
            'Climb', 'Fencing', 'Golf', 'Kick_Ball', 'Pullup',
            'Punch', 'Pushup', 'Ride_Bike', 'Ride_Horse',
            'Shoot_Ball', 'Shoot_Bow', 'Walk'
        ]

        if self.split is not None:
            search_dir = os.path.join(self.data_root, self.split)
        else:
            search_dir = self.data_root

        self.video_paths = []
        for ext in self.video_extensions:
            self.video_paths.extend(glob.glob(os.path.join(search_dir, "**", f"*{ext}"), recursive=True))

        self.video_paths.sort()

        self.video_names = [os.path.splitext(os.path.basename(path))[0] for path in self.video_paths]

        self.meta = [{'name': video_name, 'path': video_path}
                     for video_name, video_path in zip(self.video_names, self.video_paths)]

    def __len__(self):
        return len(self.meta)

    def get_meta(self, idx):
        return self.meta[idx]

    def __getitem__(self, idx):
        # This method is not actually used in the feature extraction script
        # but is required for the Dataset interface
        video_path = self.video_paths[idx]
        video, _, _ = torchvision.io.read_video(video_path)
        return video

class UCFFeatureDataset(Dataset):
    def __init__(self,
                 split=None,
                 features_root="data/ucf",
                 data_root="data/ucf/annotations",
                 max_len=300,
                 num_frames=16,
                 use_mean=False,
                 ):
        self.features_root = features_root
        self.data_root = data_root
        self.max_len = max_len
        self.num_frames = num_frames
        self.split = split
        self.all_actions = ["Climb", "Fencing", "Golf", "Kick_Ball", "Pullup", "Punch", "Pushup", "Ride_Bike", "Ride_Horse", "Shoot_Ball", "Shoot_Bow", "Walk"]

        if self.split is not None:
            annotation_path = os.path.join(self.data_root, f"{self.split}.csv")
        else:
            annotation_path = os.path.join(self.data_root, "train.csv")

        self.meta_data = []
        with open(annotation_path, 'r') as f:
            csv_reader = csv.reader(f, delimiter=',')
            for row in csv_reader:
                if len(row) == 2:  
                    video_name, action = row
                    self.meta_data.append((video_name, action))

        self.num_classes = 12

        self.video_names = [meta[0] for meta in self.meta_data]
        if use_mean:
            self.feature_paths = [os.path.join(self.features_root, f"{video_name}_mean.npy") 
                             for video_name in self.video_names]
        else:
            self.feature_paths = [os.path.join(self.features_root, f"{video_name}.npy") 
                             for video_name in self.video_names]
        
        self.labels = [int(meta[1]) for meta in self.meta_data]
        
        if len(self.feature_paths) > 0 and os.path.exists(self.feature_paths[0]):
            self.embed_dim = np.load(self.feature_paths[0]).shape[1]
        else:
            self.embed_dim = 1280
            print("Warning: No feature files found to determine embedding dimension")



    def __len__(self):
        return len(self.video_names)
    
    def __getitem__(self, idx):
        feature_path = self.feature_paths[idx]
        feature_name = self.video_names[idx]
        label_idx = self.labels[idx]

        feature = torch.from_numpy(np.load(feature_path)).float()

        label_tensor = torch.zeros(self.num_classes)
        label_tensor[label_idx] = 1.0

        T = feature.shape[0]
        if T > self.max_len:
            feature = feature[:self.max_len]
        elif T < self.max_len:
            pad_shape = (self.max_len - T,) + tuple(feature.shape[1:])
            feature = torch.cat([feature, torch.zeros(pad_shape, dtype=feature.dtype)], dim=0)
        
        mask = torch.ones(self.max_len) 
        if T < self.max_len:
            mask[T:] = 0
            
        return torch.zeros(1), feature, mask, label_tensor
        
    def calculate_map(self, predictions, targets):
        """
        Calculate accuracy for single-label classification
        Args:
            predictions: torch.Tensor of shape (N, num_classes) with predicted probabilities
            targets: torch.Tensor of shape (N, num_classes) with one-hot encoded labels
        Returns:
            accuracy score
        """
        pred_classes = torch.argmax(predictions, dim=1)
        target_classes = torch.argmax(targets, dim=1)
        
        correct = (pred_classes == target_classes).sum().item()
        total = targets.size(0)
        
        return correct / total

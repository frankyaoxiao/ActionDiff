import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision
from torchmetrics.classification import MultilabelAveragePrecision
from PIL import Image
import csv
import os
import decord
import json
import pandas as pd
import ast
import os.path as osp
import glob
import math
import matplotlib.pyplot as plt


class CharadesDataset(Dataset):
    def __init__(self,
                 split=None,
                 data_root="data/charades_ego",
                 video_path="CharadesEgo_v1/{video_name}.mp4",
                 ):
        self.data_root = data_root
        self.video_path = video_path
        self.split = split

        # random placeholder actions (replace w/ csv soon)
        self.actions = [
            'holding something', 'sitting at', 'standing at', 'watching something',
            'looking at', 'taking something from', 'putting something on',
            'opening something', 'closing something', 'walking through',
            'walking to', 'walking away from', 'picking up', 'putting down',
            'eating something', 'drinking from', 'playing with', 'using',
            'touching', 'washing', 'cleaning', 'interacting with'
        ]

        # Read metadata CSV file
        if self.split is None:
            # Load both train and test CSV files and concatenate them.
            train_csv = os.path.join(self.data_root, "annotations/CharadesEgo_v1_train.csv")
            test_csv = os.path.join(self.data_root, "annotations/CharadesEgo_v1_test.csv")
            meta_data_train = pd.read_csv(train_csv)
            meta_data_test = pd.read_csv(test_csv)
            meta_data = pd.concat([meta_data_train, meta_data_test], ignore_index=True)
        else:
            meta_path = f"annotations/CharadesEgo_v1_{self.split}.csv"
            meta_data = pd.read_csv(os.path.join(self.data_root, meta_path))

        meta_data = meta_data.sort_values('id').to_dict(orient='records')

        # Extract video IDs and create full paths
        self.video_names = [meta['id'] for meta in meta_data]
        self.video_paths = [os.path.join(self.data_root,
                                          self.video_path.format(video_name=video_name))
                            for video_name in self.video_names]

        self.meta = [{'name': video_name, 'path': video_path}
                     for video_name, video_path in zip(self.video_names, self.video_paths)]

    def __len__(self):
        return len(self.meta)

    def get_meta(self, idx):
        return self.meta[idx]

    def __getitem__(self, idx):
        # Never actually used
        video_path = self.video_paths[idx]
        video_name = self.video_names[idx]
        video, _, _ = torchvision.io.read_video(video_path)
        return video

class CharadesFeatureDataset(Dataset):
    def __init__(self,
                 split=None,
                 viewpoint=None,
                 features_root="data/charadesego/20",
                 meta_path="data/charadesego/annotations",
                 classes_path="data/charadesego/annotations/Charades_v1_classes.txt",
                 truncate_mode=None,
                 max_len=300,
                 num_frames=16,
                 use_mean=True,
                 ):
        self.features_root = features_root
        self.classes_path = classes_path
        self.max_len = max_len
        self.num_frames = num_frames
        self.split = split
        self.viewpoint = viewpoint
        self.use_mean = use_mean

        filepath = "CharadesEgo_v1"
        if split is not None:
            filepath += f"_{split}"
        if viewpoint is not None:
            if viewpoint == 1:
                filepath += "_only1st"
            elif viewpoint == 3:
                filepath += "_only3rd"
        filepath += ".csv"
        self.meta_path = os.path.join(meta_path, filepath)

        all_meta_data = []
        with open(self.meta_path, 'r') as f:
            csvFile = csv.reader(f)
            next(csvFile)
            for lines in csvFile:
                all_meta_data.append(lines)

        with open(self.classes_path, 'r') as f:
            classes = f.readlines()
        self.all_actions = [line.strip()[5:] for line in classes]
        self.num_classes = len(self.all_actions)

        self.video_names = [meta[0] for meta in all_meta_data]
        if not self.use_mean:
            self.feature_paths = [os.path.join(self.features_root, f"{video_name}.npy") for video_name in self.video_names]
        else:
            self.feature_paths = [os.path.join(self.features_root, f"{video_name}_mean.npy") for video_name in self.video_names]

        self.labels = []
        for i in all_meta_data:
            temp_labels = []
            labels = i[9].strip().split(';')
            for label in labels:
                if label:  
                    temp_labels.append(label[1:4])
            self.labels.append(temp_labels)
            
        if self.feature_paths and os.path.exists(self.feature_paths[0]):
            self.embed_dim = np.load(self.feature_paths[0]).shape[1]
        else:
            self.embed_dim = 1280
            print("Warning: No feature files found to determine embedding dimension")
        

    def __len__(self):
        return len(self.video_names)
    
    def __getitem__(self, idx):
        feature_path = self.feature_paths[idx]
        feature_name = self.video_names[idx]
        labels = self.labels[idx]

        feature = torch.from_numpy(np.load(feature_path)).float()
        label_tensor = torch.zeros(self.num_classes)
        for label in labels:
            if label != '':
                label_tensor[int(label)] = 1.0

        T = feature.shape[0]
        if T > self.max_len:
            feature = feature[:self.max_len]
        elif T < self.max_len:
            pad_shape = (self.max_len - T,) + tuple(feature.shape[1:])
            feature = torch.cat([feature, torch.zeros(pad_shape, dtype=feature.dtype)], dim=0)
        
        # Create attention mask (1 for real tokens, 0 for padding)
        mask = torch.ones(self.max_len) 
        if T < self.max_len:
            mask[T:] = 0
            
        return torch.zeros(1), feature, mask, label_tensor
        
    def calculate_map(self, predictions, targets):
        """
        Calculate mean Average Precision using torchmetrics
        Args:
            predictions: torch.Tensor of shape (N, num_classes) with predicted probabilities
            targets: torch.Tensor of shape (N, num_classes) with binary labels
        Returns:
            mAP score
        """
        
        metric = MultilabelAveragePrecision(num_labels=len(self.all_actions), average='micro')
        
        predictions = predictions.float()
        targets = targets.int()
        
        map_score = metric(predictions, targets)
        
        return map_score.item()




        
        
class CharadesFeatureDatasetSplit(Dataset):
    def __init__(self, 
                 split='train', 
                 features_root='data/video_feats', 
                 max_len=None, 
                 viewpoint=None,
                 classes_path="data/charadesego/annotations/Charades_v1_classes.txt"):
        self.split = split
        self.features_root = features_root
        self.max_len = max_len
        self.viewpoint = viewpoint
        self.feature_stride = 4
        self.classes_path = classes_path
        
        with open(self.classes_path, 'r') as f:
            classes = f.readlines()
        self.all_actions = [line.strip()[5:] for line in classes]
        self.num_classes = len(self.all_actions)
        
        # Load annotations
        if split == 'train':
            csv_path = 'data/charadesego/annotations/CharadesEgo_v1_train_only3rd_SPLIT.csv'
        else:
            csv_path = 'data/charadesego/annotations/CharadesEgo_v1_test_only3rd.csv'
        
        self.df = pd.read_csv(csv_path)
        
        # Filter out invalid time ranges where end_time <= start_time
        valid_times = self.df['end'].astype(float) > self.df['start'].astype(float)
        invalid_count = (~valid_times).sum()
        if invalid_count > 0:
            print(f"Warning: Filtered out {invalid_count} samples where end_time <= start_time")
        self.df = self.df[valid_times].reset_index(drop=True)
        
        video_dir = 'data/charadesego/videos'
        video_list = glob.glob(osp.join(video_dir, '*.mp4'))
        self.fps_dict = {}
        for video in video_list:
            video_id = osp.basename(video).split('.')[0]
            try:
                vr = decord.VideoReader(video)
                self.fps_dict[video_id] = vr.get_avg_fps()
            except:
                print(f"Warning: Could not read video {video}")
                continue
                
        valid_videos = set(self.fps_dict.keys())
        self.df = self.df[self.df['id'].isin(valid_videos)].reset_index(drop=True)
        
        if len(self.df) == 0:
            raise ValueError("No valid videos found in the dataset")
        self.embed_dim = 1280

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_id = row['id']
        start_time = float(row['start'])
        end_time = float(row['end'])
        action = row['action']
        
        feature_path = osp.join(self.features_root, f"{video_id}_mean.npy")
        if not osp.exists(feature_path):
            raise FileNotFoundError(f"Feature not found: {feature_path}")
        
        feature = np.load(feature_path, allow_pickle=True).item()[3]
        
        fps = self.fps_dict.get(video_id)
        if fps is None:
            raise ValueError(f"FPS not found for video {video_id}")
            
        start_frame = math.floor((start_time * fps) / self.feature_stride) - 2
        end_frame = math.ceil((end_time * fps) / self.feature_stride) + 2
        
        start_frame = max(0, start_frame)
        end_frame = min(feature.shape[0], end_frame)
        
        feature = feature[start_frame:end_frame]
        
        feature = torch.from_numpy(feature).float()
        
        mask = torch.ones(feature.shape[0], dtype=torch.bool)
        
        if self.max_len is not None:
            if feature.shape[0] > self.max_len:
                feature = feature[:self.max_len]
                feature = feature[:self.max_len]
                mask = mask[:self.max_len]
            elif feature.shape[0] < self.max_len:
                mask = torch.cat([mask, torch.zeros(self.max_len - feature.shape[0], dtype=torch.bool)])
                pad_length = self.max_len - feature.shape[0]
                feature = torch.nn.functional.pad(feature, (0, 0, 0, pad_length))
        
        label = torch.zeros(self.num_classes)
        try:
            action_idx = self.all_actions.index(action)
            label[action_idx] = 1
        except ValueError:
            print(f"Warning: Action '{action}' not found in classes list")
        
        return torch.zeros(1), feature, mask, label





        
        
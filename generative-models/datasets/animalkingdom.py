import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchmetrics.classification import MultilabelAveragePrecision
import torchvision
from PIL import Image

import glob
import os
import json
import time

import numpy as np
import pandas as pd
import ast
import matplotlib.pyplot as plt

class AnimalKingdomNewActionsDataset(Dataset):
    def __init__(self,
                 split=None,
                 features_root="data/video_feats_new/video_feats_fp32/feats_layer_idxs=3",
                 meta_path="data/animal_kingdom/annotations/AR_metadata.xlsx",
                 target_animals=[
                    "Bird", 
                     "Fish", 
                     "Frog / Toad", 
                     "Snake / Cobra / Viper / Python",
                     "Lizard / Gecko / Draco / Iguana / Chamelon",
                     "Primate / Monkey / Macaque / Baboon / Chimpanzee / Gorilla / Orangutan / Langur",
                     "Spider / Tarantula",
                     "Cricket / Grasshopper / Praying mantis / Leafhopper",
                     "Water bird / Duck / Swan / Goose"
                 ],
                 train_actions=["Eating", "Attending", "Moving"],
                 test_actions=["Swimming", "Sensing", "Keeping Still"],
                 use_mean=True,
                 max_len=100,
                 num_frames=16):
        self.features_root = features_root
        self.meta_path = meta_path
        self.split = split
        self.use_mean = use_mean
        self.num_frames = num_frames
        self.max_len = max_len

        self.target_animals = [animal.lower() for animal in target_animals]
        self.train_actions = [action.lower() for action in train_actions]
        self.test_actions = [action.lower() for action in test_actions]

        # Create a unified action list that combines both train and test actions
        self.all_actions = self.train_actions + self.test_actions
        
        # Create a mapping from action to index that's consistent across train/test
        self.action_to_idx = {action: idx for idx, action in enumerate(self.all_actions)}

        meta_data_all = pd.read_excel(self.meta_path)
        meta_data_all = meta_data_all.to_dict(orient='records')

        meta_data = []
        if split is not None:
            if split == 'train':
                for meta in meta_data_all:
                    formatted = read_list(meta['list_animal_action'])
                    labels_formatted = read_list(meta['labels'])
                    if len(labels_formatted) == 1:
                        animal = formatted[0][0].lower()
                        if animal == "common quail":
                            animal = "common quail bird"
                        elif animal == "cuckoo bird":
                            animal = "common cuckoo bird"
                        action = formatted[0][1].lower()
                        if animal not in self.target_animals:
                            continue
                        if action in self.train_actions:
                            meta_data.append(meta)
            elif split == 'test':
                for meta in meta_data_all:
                    formatted = read_list(meta['list_animal_action'])
                    labels_formatted = read_list(meta['labels'])
                    if len(labels_formatted) == 1:
                        animal = formatted[0][0].lower()
                        if animal == "common quail":
                            animal = "common quail bird"
                        elif animal == "cuckoo bird":
                            animal = "common cuckoo bird"
                        action = formatted[0][1].lower()
                        if animal not in self.target_animals:
                            continue
                        if action in self.test_actions:
                            meta_data.append(meta)
            else:
                raise ValueError(f"Split {split} not recognized. Use 'train' or 'test'")
        print("Amount of data", len(meta_data))

        temp_feature_names = [meta['video_id'] for meta in meta_data]
        if use_mean:
            temp_feature_paths = [os.path.join(self.features_root, f"{feature_name}_mean.npy") for feature_name in temp_feature_names]
        else:
            temp_feature_paths = [os.path.join(self.features_root, f"{feature_name}.npy") for feature_name in temp_feature_names]
        temp_feature_vids = [os.path.join("data", "image", feature_name) for feature_name in temp_feature_names]
        self.feature_names = []
        self.feature_paths = []
        self.feature_videos = []
        for name, path, vid in zip(temp_feature_names, temp_feature_paths, temp_feature_vids):
            self.feature_names.append(name)
            self.feature_paths.append(path)
            files = sorted(os.listdir(vid))
            self.feature_videos.append(files)
        self.meta = [{'name': feature_name, 'path': feature_path} for feature_name, feature_path in zip(self.feature_names, self.feature_paths)]

        self.annots = {}
        for meta in meta_data:
            name = meta['video_id']
            labels = read_list(meta['labels'])
            animal_actions = read_list(meta['list_animal_action'])
            animal = animal_actions[0][0].lower()
            if animal == "common quail":
                animal = "common quail bird"
            elif animal == "cuckoo bird":
                animal = "common cuckoo bird"
            action = animal_actions[0][1].lower()
            self.annots[name] = (labels, animal_actions, animal)

        self.all_actions_numerical = set()
        for meta in meta_data_all:  
            labels = read_list(meta['labels'])
            self.all_actions_numerical.update(labels)
        self.all_actions_numerical = sorted([int(label) for label in self.all_actions_numerical])
        self.label2idx = {str(label): idx for idx, label in enumerate(self.all_actions_numerical)}
        
        # Set num_classes to the number of actions in our combined list
        self.num_classes = len(self.all_actions)
        
        if len(self.feature_paths) > 0:
            self.embed_dim = np.load(self.feature_paths[0]).shape[1]
        else:
            self.embed_dim = 1280  # Default value

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        feature_path = self.feature_paths[idx]
        feature_name = self.feature_names[idx]
        images = torch.zeros(1)
        feature = torch.from_numpy(np.load(feature_path)).float()
            
        # Create a label tensor with a consistent mapping
        label_tensor = torch.zeros(self.num_classes)
        labels, animal_actions, animal = self.annots[feature_name]
        
        # Get the action from animal_actions and find its index in our unified mapping
        action = animal_actions[0][1].lower()
        action_idx = self.action_to_idx[action]
        
        # Set the corresponding position in the one-hot tensor
        label_tensor[action_idx] = 1.0

        T = feature.shape[0]
        if T > self.max_len:
            feature = feature[:self.max_len]
        elif T < self.max_len:
            pad_shape = (self.max_len - T,) + tuple(feature.shape[1:])
            feature = torch.cat([feature, torch.zeros(pad_shape, dtype=feature.dtype)], dim=0)
        
        mask = torch.ones(self.max_len)
        if T < self.max_len:
            mask[T:] = 0

        return images, feature, mask, label_tensor
        
    def calculate_map(self, predictions, targets):
        """
        Calculate accuracy by comparing max prediction with target
        Args:
            predictions: torch.Tensor of shape (N, num_classes) with predicted probabilities
            targets: torch.Tensor of shape (N, num_classes) with one-hot encoded labels
        Returns:
            accuracy score
        """
        pred_indices = torch.argmax(predictions, dim=1)
        print(pred_indices)
        target_indices = torch.argmax(targets, dim=1)
        correct = (pred_indices == target_indices).sum().item()
        total = targets.size(0)
        
        return correct / total


class AnimalKingdomNewSpeciesDataset(Dataset):
    def __init__(self,
                 split=None,
                 features_root="data/video_feats_new/video_feats_fp32/feats_layer_idxs=3",
                 meta_path="data/animal_kingdom/annotations/AR_metadata.xlsx",
                 target_actions=["Moving", "Eating","Attending", "Swimming", "Sensing", "Keeping Still"],
                 train_animals=["Bird", "Fish", "Frog / Toad", "Snake / Cobra / Viper / Python"],
                 test_animals=["Lizard / Gecko / Draco / Iguana / Chamelon", "Primate / Monkey / Macaque / Baboon / Chimpanzee / Gorilla / Orangutan / Langur",
                               "Spider / Tarantula", "Cricket / Grasshopper / Praying mantis / Leafhopper", "Water bird / Duck / Swan / Goose"],
                 use_mean=True,
                 max_len=100,
                 num_frames=16):
        self.features_root = features_root
        self.meta_path = meta_path
        self.split = split
        self.use_mean = use_mean
        self.num_frames = num_frames
        
        # Convert all actions to lowercase
        self.all_actions = [action.lower() for action in [
            'Keeping still', 'Dancing', 'Doing A Neck Raise', 'Standing', 'Performing allo-grooming',
            'Swimming', 'Jumping', 'Swimming In Circles', 'Walking', 'Chirping', 'Shaking Head',
            'Sensing', 'Eating', 'Carrying', 'Performing sexual pursuit', 'Hatching', 'Escaping',
            'Leaning', 'Trapped', 'Rubbing its head', 'Disturbing Another Animal', 'Sharing Food',
            'Dancing On Water', 'Sleeping', 'Defensive Rearing', 'Detaching As A Parasite', 'Resting',
            'Falling', 'Walking On Water', 'Exiting Nest', 'Unrolling', 'Doing Push Up',
            'Doing somersault', 'Digging', 'Dead', 'Landing', 'Pecking', 'Hissing', 'Biting',
            'Urinating', 'Retaliating', 'Doing A Back Kick', 'Dying', 'Exiting Cocoon',
            'Performing copulatory mounting', 'Stinging', 'Drinking', 'Washing', 'Calling',
            'Having A Flehmen Response', 'Standing in alert', 'Climbing', 'Lying On Top',
            'Competing For Dominance', 'Lying on its side', 'Gliding', 'Coiling', 'Sinking', 'Rolling',
            'Spitting', 'Giving Birth', 'Molting', 'Fighting', 'Flapping its ears', 'Holding Hands',
            'Carrying In Mouth', 'Displaying Defensive Pose', 'Camouflaging',
            'Performing allo-preening', 'Performing sexual exploration', 'Swaying', 'Spreading',
            'Drifting', 'Turning Around', 'Wrapping Itself Around Prey', 'Hanging', 'Preying',
            'Panting', 'Doing A Backward Tilt', 'Being Eaten', 'Manipulating Object', 'Building nest',
            'Grooming', 'Spreading Wings', 'Pounding', 'Struggling', 'Immobilized', 'Startled',
            'Playing', 'Abseiling', 'Preening', 'Hugging', 'Hopping', 'Running', 'Running On Water',
            'Attending', 'Rattling', 'Pulling', 'Yawning', 'Licking', 'Unmounting', 'Retreating',
            'Swinging', 'Wrapping Prey', 'Playing Dead', 'Flapping', 'Being Carried In Mouth',
            'Flapping Tail', 'Fleeing', 'Waving', 'Performing Sexual Display', 'Showing Affection',
            'Diving', 'Defecating', 'Being Dragged', 'Exploring', 'Laying Eggs', 'Entering its nest',
            'Surfacing', 'Doing A Chin Dip', 'Barking', 'Sitting', 'Getting Bullied', 'Tail Swishing',
            'Undergoing Chrysalis', 'Doing A Face Dip', 'Chasing', 'Shaking', 'Squatting', 'Attacking',
            'Moving', 'Spitting Venom', 'Puffing its throat', 'Gasping For Air', 'Doing A Side Tilt',
            'Being Carried', 'Sleeping in its nest', 'Flying', 'Giving Off Light', 'Lying Down'
        ]]
        
        self.target_actions = [action.lower() for action in target_actions]
        self.train_animals = [animal.lower() for animal in train_animals]
        self.test_animals = [animal.lower() for animal in test_animals]

        meta_data_all = pd.read_excel(self.meta_path)
        meta_data_all = meta_data_all.to_dict(orient='records')

        species_data = pd.read_excel(self.meta_path, sheet_name=3)
        species_data = species_data.to_dict(orient='records')
        # Convert species dictionary keys and values to lowercase
        species_dict = {species['Animal'].lower(): species['Sub-Class'].lower() for species in species_data}

        meta_data = []
        if split is not None:
            if split == 'train':
                for meta in meta_data_all:
                    formatted = read_list(meta['list_animal_action'])
                    labels_formatted = read_list(meta['labels'])
                    #if animal in species_dict and len(labels_formatted) == 1:
                    #    print("found one", formatted[0][1])
                    if len(labels_formatted) == 1:
                        animal = formatted[0][0].lower()
                        if animal == "common quail":
                            animal = "common quail bird"
                        elif animal == "cuckoo bird":
                            animal = "common cuckoo bird"
                        action = formatted[0][1].lower()
                        #print(action, self.target_actions)
                        #print(species_dict[animal], self.train_animals)
                        #if (species_dict[animal] in self.train_animals):
                        #    print(animal, species_dict[animal])
                        if animal not in species_dict:
                            print(f"Skipping unknown animal: {formatted[0][0]}")
                            continue
                        if species_dict[animal] in self.train_animals and action in self.target_actions:
                            #print(animal, action)
                            #print(formatted)
                            meta_data.append(meta)
            elif split == 'test':
                for meta in meta_data_all:
                    formatted = read_list(meta['list_animal_action'])
                    labels_formatted = read_list(meta['labels'])
                    if len(labels_formatted) == 1:
                        animal = formatted[0][0].lower()
                        if animal == "common quail":
                            animal = "common quail bird"
                        elif animal == "cuckoo bird":
                            animal = "common cuckoo bird"
                        action = formatted[0][1].lower()
                        if animal not in species_dict:
                            print(f"Skipping unknown animal: {formatted[0][0]}")
                            continue
                        if species_dict[animal] in self.test_animals and action in self.target_actions:
                            #print(animal, action)
                            meta_data.append(meta)
            else:
                raise ValueError(f"Split {split} not recognized. Use 'train' or 'test'")
        
        print(self.train_animals)
        print(self.test_animals)
        print("Amount of data", len(meta_data))
        
        temp_feature_names = [meta['video_id'] for meta in meta_data]
        if use_mean:
            temp_feature_paths = [os.path.join(self.features_root, f"{feature_name}_mean.npy") for feature_name in temp_feature_names]
        else:
            temp_feature_paths = [os.path.join(self.features_root, f"{feature_name}.npy") for feature_name in temp_feature_names]

        temp_feature_vids = [os.path.join("data", 'image', feature_name) for feature_name in temp_feature_names]
        
        features = []
        self.feature_names = []
        self.feature_paths = []
        self.feature_videos = []
        for name, path, vid in zip(temp_feature_names, temp_feature_paths, temp_feature_vids):
            self.feature_names.append(name)
            self.feature_paths.append(path)
            #files = sorted(os.listdir(vid))
            #self.feature_videos.append(files)

        self.meta = [{'name': feature_name, 'path': feature_path} for feature_name, feature_path in zip(self.feature_names, self.feature_paths)]


        self.annots = {}
        for meta in meta_data:  
            name = meta['video_id']
            labels = read_list(meta['labels'])
            animal_actions = read_list(meta['list_animal_action'])
            animal = read_list(meta['list_animal_action'])[0][0].lower()
            if animal == "common quail":
                animal = "common quail bird"
            elif animal == "cuckoo bird":
                animal = "common cuckoo bird"
            subclass = species_dict[animal]
            self.annots[name] = (labels, animal_actions, subclass)

        self.max_len = max_len

        self.all_actions_numerical = set()
        for meta in meta_data_all:  
            labels = read_list(meta['labels'])
            self.all_actions_numerical.update(labels)
        self.all_actions_numerical = sorted([int(label) for label in self.all_actions_numerical])
        self.label2idx = {str(label): idx for idx, label in enumerate(self.all_actions_numerical)}
        self.num_classes = len(self.all_actions_numerical)
        self.embed_dim = np.load(self.feature_paths[0]).shape[1]

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        feature_path = self.feature_paths[idx]
        feature_name = self.feature_names[idx]
        #feature_images = self.feature_videos[idx]
        labels, animal_actions, subclass = self.annots[feature_name]
        
        images = torch.zeros(1)
        feature = torch.from_numpy(np.load(feature_path)).float()
        
        label_tensor = torch.zeros(self.num_classes)
        for label in labels:
            str_label = str(label)
            if str_label not in self.label2idx:
                raise KeyError(f"Label {label} (type: {type(label)}) not found in mapping. "
                             f"Available labels: {sorted(list(self.label2idx.keys()))[:10]}...")
            label_tensor[self.label2idx[str_label]] = 1.0
        
        # Pad or truncate to max_len
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
            
        return images, feature, mask, label_tensor#, subclass, animal_actions[0][1]

    def _sample_indices(self, num_frames):
        if num_frames <= self.num_frames:
            indices = np.linspace(0, num_frames - 1, self.num_frames, dtype=int)
        else:
            ticks = np.linspace(0, num_frames, self.num_frames + 1, dtype=int)
            indices = ticks[:-1] + (ticks[1:] - ticks[:-1]) // 2
        return indices

    def _load_image(self, directory, image_name):
        image_path = os.path.join(directory, image_name)
        image = torchvision.io.read_image(image_path)  
        
        image = torchvision.transforms.functional.resize(
            image,
            (224, 224),
            interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
            antialias=True
        )
        
        image = image.float() / 255.0
        
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(-1, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(-1, 1, 1)
        image = (image - mean) / std
        
        return image

    def get_action_names(self):
        """
        Returns list of action names corresponding to numerical labels
        """
        return [self.all_actions[idx] for idx in self.all_actions_numerical]

    def calculate_map(self, predictions, targets):
        """
        Calculate accuracy by comparing max prediction with target
        Args:
            predictions: torch.Tensor of shape (N, num_classes) with predicted probabilities
            targets: torch.Tensor of shape (N, num_classes) with binary labels
        Returns:
            accuracy score
        """
        pred_indices = torch.argmax(predictions, dim=1)
        target_indices = torch.argmax(targets, dim=1)
        
        correct = (pred_indices == target_indices).sum().item()
        total = targets.size(0)
        
        return correct / total


class AnimalKingdomFeatureDataset(Dataset):
    def __init__(self,
                 split=None,
                 features_root="data/video_feats_new/video_feats_fp32/feats_layer_idxs=3",
                 meta_path="data/animal_kingdom/annotations/AR_metadata.xlsx",
                 use_mean=True,
                 use_all=False,
                 max_len=100,
                 return_meta=False,
                 return_video=False,
                 videos_root="data/animal_kingdom/video",
                 num_frames=16,
                 truncate_mode='beg',
                 stride=1):
        self.return_meta = return_meta
        self.return_video = return_video
        self.videos_root = videos_root
        self.truncate_mode = truncate_mode
        self.stride = stride
        self.features_root = features_root
        self.meta_path = meta_path
        self.split = split
        self.num_frames = num_frames
        self.use_mean = use_mean
        self.use_all = use_all
        self.max_len = max_len
        self.all_actions = [
        'Keeping still', 'Dancing', 'Doing A Neck Raise', 'Standing', 'Performing allo-grooming',
        'Swimming', 'Jumping', 'Swimming In Circles', 'Walking', 'Chirping', 'Shaking Head',
        'Sensing', 'Eating', 'Carrying', 'Performing sexual pursuit', 'Hatching', 'Escaping',
        'Leaning', 'Trapped', 'Rubbing its head', 'Disturbing Another Animal', 'Sharing Food',
        'Dancing On Water', 'Sleeping', 'Defensive Rearing', 'Detaching As A Parasite', 'Resting',
        'Falling', 'Walking On Water', 'Exiting Nest', 'Unrolling', 'Doing Push Up',
        'Doing somersault', 'Digging', 'Dead', 'Landing', 'Pecking', 'Hissing', 'Biting',
        'Urinating', 'Retaliating', 'Doing A Back Kick', 'Dying', 'Exiting Cocoon',
        'Performing copulatory mounting', 'Stinging', 'Drinking', 'Washing', 'Calling',
        'Having A Flehmen Response', 'Standing in alert', 'Climbing', 'Lying On Top',
        'Competing For Dominance', 'Lying on its side', 'Gliding', 'Coiling', 'Sinking', 'Rolling',
        'Spitting', 'Giving Birth', 'Molting', 'Fighting', 'Flapping its ears', 'Holding Hands',
        'Carrying In Mouth', 'Displaying Defensive Pose', 'Camouflaging',
        'Performing allo-preening', 'Performing sexual exploration', 'Swaying', 'Spreading',
        'Drifting', 'Turning Around', 'Wrapping Itself Around Prey', 'Hanging', 'Preying',
        'Panting', 'Doing A Backward Tilt', 'Being Eaten', 'Manipulating Object', 'Building nest',
        'Grooming', 'Spreading Wings', 'Pounding', 'Struggling', 'Immobilized', 'Startled',
        'Playing', 'Abseiling', 'Preening', 'Hugging', 'Hopping', 'Running', 'Running On Water',
        'Attending', 'Rattling', 'Pulling', 'Yawning', 'Licking', 'Unmounting', 'Retreating',
        'Swinging', 'Wrapping Prey', 'Playing Dead', 'Flapping', 'Being Carried In Mouth',
        'Flapping Tail', 'Fleeing', 'Waving', 'Performing Sexual Display', 'Showing Affection',
        'Diving', 'Defecating', 'Being Dragged', 'Exploring', 'Laying Eggs', 'Entering its nest',
        'Surfacing', 'Doing A Chin Dip', 'Barking', 'Sitting', 'Getting Bullied', 'Tail Swishing',
        'Undergoing Chrysalis', 'Doing A Face Dip', 'Chasing', 'Shaking', 'Squatting', 'Attacking',
        'Moving', 'Spitting Venom', 'Puffing its throat', 'Gasping For Air', 'Doing A Side Tilt',
        'Being Carried', 'Sleeping in its nest', 'Flying', 'Giving Off Light', 'Lying Down'
    ]

        meta_data_all = pd.read_excel(self.meta_path)
        meta_data_all = meta_data_all.to_dict(orient='records')

        if split is not None:
            if split == 'train':
                meta_data = [meta for meta in meta_data_all if meta['type'] == 'train']
            elif split == 'test':
                meta_data = [meta for meta in meta_data_all if meta['type'] == 'test']
            else:
                raise ValueError(f"Split {split} not recognized. Use 'train' or 'test'")
        
        temp_feature_names = [meta['video_id'] for meta in meta_data]
        temp_feature_vids = [os.path.join("data", 'image', feature_name) for feature_name in temp_feature_names]
        if use_mean:
            temp_feature_paths = [os.path.join(self.features_root, f"{feature_name}_mean.npy") for feature_name in temp_feature_names]
        else:
            temp_feature_paths = [os.path.join(self.features_root, f"{feature_name}.npy") for feature_name in temp_feature_names]
        
        features = []
        self.feature_names = []
        self.feature_paths = []
        self.feature_videos = [0] * len(temp_feature_names) 
        self.video_paths = [0] * len(temp_feature_names) 
        for name, path, vid in zip(temp_feature_names, temp_feature_paths, temp_feature_vids):
            self.feature_names.append(name)
            self.feature_paths.append(path)
            #files = sorted(os.listdir(vid))
            #self.feature_videos.append(files)
            #self.video_paths.append(os.path.join(self.videos_root, f'{name}.mp4'))

        #print(features)
        #print("SKIPPED: ", len(features))
        self.meta = [{'name': feature_name, 'path': feature_path} for feature_name, feature_path in zip(self.feature_names, self.feature_paths)]


        self.annots = {}
        for meta in meta_data_all:
            name = meta['video_id']
            labels = read_list(meta['labels'])
            animal_actions = read_list(meta['list_animal_action'])
            self.annots[name] = (labels, animal_actions)


        self.all_actions_numerical = set()
        for labels, _ in self.annots.values():
            self.all_actions_numerical.update(labels)
        self.all_actions_numerical = sorted([int(label) for label in self.all_actions_numerical])
        self.label2idx = {str(label): idx for idx, label in enumerate(self.all_actions_numerical)}
        self.num_classes = len(self.all_actions_numerical)
        self.embed_dim = np.load(self.feature_paths[0]).shape[1]
        
        # Create distribution visualization
        self.create_data_distribution_visualization(meta_data_all)

    def create_data_distribution_visualization(self, meta_data_all):
        """Create and save a histogram of the dataset distribution."""
        # Create figures directory if it doesn't exist
        os.makedirs(os.path.join(os.path.dirname(self.meta_path), 'figures'), exist_ok=True)
        
        # Count samples in each split
        train_count = 0
        test_count = 0
        
        # Count unique actions in each split
        train_actions = set()
        test_actions = set()
        
        # Count unique animals in each split
        train_animals = set()
        test_animals = set()
        
        for meta in meta_data_all:
            split_type = meta.get('type', 'unknown')
            animal_actions = read_list(meta['list_animal_action'])
            
            if split_type == 'train':
                train_count += 1
                for animal_action in animal_actions:
                    if len(animal_action) >= 2:
                        animal, action = animal_action[0].lower(), animal_action[1].lower()
                        train_animals.add(animal)
                        train_actions.add(action)
            elif split_type == 'test':
                test_count += 1
                for animal_action in animal_actions:
                    if len(animal_action) >= 2:
                        animal, action = animal_action[0].lower(), animal_action[1].lower()
                        test_animals.add(animal)
                        test_actions.add(action)
        
        # Create distribution plots
        plt.figure(figsize=(12, 6))
        
        # Main plot - total sample counts
        counts = [train_count, test_count]
        plt.bar(['Train', 'Test'], counts, color=['#3498db', '#e74c3c'])
        
        # Add count labels
        for i, count in enumerate(counts):
            plt.text(i, count + 5, str(count), ha='center', fontsize=12)
        
        plt.title('Animal Kingdom Dataset Distribution', fontsize=16)
        plt.ylabel('Number of Videos', fontsize=14)
        plt.ylim(0, max(counts) * 1.15)
        
        # Save the main plot
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(self.meta_path), 'figures', 'dataset_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create a plot showing unique actions/animals counts
        plt.figure(figsize=(12, 6))
        
        # Set up data
        categories = ['Unique Actions', 'Unique Animals']
        train_unique = [len(train_actions), len(train_animals)]
        test_unique = [len(test_actions), len(test_animals)]
        
        # Set up bar positions
        x = np.arange(len(categories))
        width = 0.35
        
        # Create bars
        plt.bar(x - width/2, train_unique, width, label='Train', color='#3498db')
        plt.bar(x + width/2, test_unique, width, label='Test', color='#e74c3c')
        
        # Add labels
        for i, count in enumerate(train_unique):
            plt.text(i - width/2, count + 1, str(count), ha='center', fontsize=12)
        
        for i, count in enumerate(test_unique):
            plt.text(i + width/2, count + 1, str(count), ha='center', fontsize=12)
        
        plt.title('Unique Actions and Animals by Split', fontsize=16)
        plt.ylabel('Count', fontsize=14)
        plt.xticks(x, categories)
        plt.legend()
        
        # Save the unique counts plot
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(self.meta_path), 'figures', 'unique_categories.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create histograms for action distribution
        # Count action frequencies
        train_action_counts = {}
        test_action_counts = {}
        
        # Count animal frequencies 
        train_animal_counts = {}
        test_animal_counts = {}
        
        for meta in meta_data_all:
            split_type = meta.get('type', 'unknown')
            animal_actions = read_list(meta['list_animal_action'])
            
            for animal_action in animal_actions:
                if len(animal_action) >= 2:
                    animal = animal_action[0].lower()
                    action = animal_action[1].lower()
                    
                    if split_type == 'train':
                        if action not in train_action_counts:
                            train_action_counts[action] = 0
                        train_action_counts[action] += 1
                        
                        if animal not in train_animal_counts:
                            train_animal_counts[animal] = 0
                        train_animal_counts[animal] += 1
                        
                    elif split_type == 'test':
                        if action not in test_action_counts:
                            test_action_counts[action] = 0
                        test_action_counts[action] += 1
                        
                        if animal not in test_animal_counts:
                            test_animal_counts[animal] = 0
                        test_animal_counts[animal] += 1
        
        plt.figure(figsize=(14, 8))
        
        train_action_values = list(train_action_counts.values())
        test_action_values = list(test_action_counts.values())
        
        min_count = min(min(train_action_values) if train_action_values else 1, 
                       min(test_action_values) if test_action_values else 1)
        max_count = max(max(train_action_values) if train_action_values else 1, 
                       max(test_action_values) if test_action_values else 1)
        
        if min_count < 1:
            min_count = 1  
        
        bins = np.logspace(np.log10(min_count), np.log10(max_count + 1), 15)
        
        plt.hist([train_action_values, test_action_values], bins=bins, 
                label=['Train', 'Test'], color=['#3498db', '#e74c3c'], alpha=0.7)
        
        plt.title('Distribution of Action Frequencies', fontsize=16)
        plt.xlabel('Frequency of Action (log scale)', fontsize=14)
        plt.ylabel('Number of Action Categories', fontsize=14)
        plt.xscale('log')  
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        plt.grid(axis='x', which='major', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(self.meta_path), 'figures', 'action_histogram.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        plt.figure(figsize=(14, 8))
        
        train_animal_values = list(train_animal_counts.values())
        test_animal_values = list(test_animal_counts.values())
        
        min_count = min(min(train_animal_values) if train_animal_values else 1, 
                       min(test_animal_values) if test_animal_values else 1)
        max_count = max(max(train_animal_values) if train_animal_values else 1, 
                       max(test_animal_values) if test_animal_values else 1)
        
        if min_count < 1:
            min_count = 1  
            
        bins = np.logspace(np.log10(min_count), np.log10(max_count + 1), 15)
        
        plt.hist([train_animal_values, test_animal_values], bins=bins, 
                label=['Train', 'Test'], color=['#3498db', '#e74c3c'], alpha=0.7)
        
        plt.title('Distribution of Animal Frequencies', fontsize=16)
        plt.xlabel('Frequency of Animal (log scale)', fontsize=14)
        plt.ylabel('Number of Animal Categories', fontsize=14)
        plt.xscale('log')  
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        plt.grid(axis='x', which='major', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(self.meta_path), 'figures', 'animal_histogram.png'), dpi=300, bbox_inches='tight')
        
        print(f"Dataset distribution visualizations saved to {os.path.join(os.path.dirname(self.meta_path), 'figures/')}")

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        feature_path = self.feature_paths[idx]
        feature_name = self.feature_names[idx]
        #feature_images = self.feature_videos[idx]
        labels = self.annots[feature_name][0]
        
        #indices = self._sample_indices(len(feature_images))
        #images = [self._load_image(os.path.join("data", "image", feature_name), feature_images[i]) for i in indices]
        #images = torch.stack(images)
        images = torch.zeros(1)

        if self.return_video:
            video_path = self.video_paths[idx]
            feature, _, _ = torchvision.io.read_video(video_path)  # T x H x W x C
        else:
            feature = torch.from_numpy(np.load(feature_path)).float()

        feature = feature[::self.stride]
        
        label_tensor = torch.zeros(self.num_classes)
        for label in labels:
            str_label = str(label)
            if str_label not in self.label2idx:
                raise KeyError(f"Label {label} (type: {type(label)}) not found in mapping. "
                             f"Available labels: {sorted(list(self.label2idx.keys()))[:10]}...")
            label_tensor[self.label2idx[str_label]] = 1.0

        # Create attention mask (1 for real tokens, 0 for padding)
        mask = torch.ones(self.max_len)

        # Pad or truncate to max_len
        T = feature.shape[0]
        if T > self.max_len:
            if self.truncate_mode == 'beg':
                feature = feature[:self.max_len]
            elif self.truncate_mode == 'mid':
                mid = T // 2,
                left = mid - self.max_len // 2,
                right = left + self.max_len
                feature = feature[left:right]
            elif self.truncate_mode == 'end':
                feature = feature[-self.max_len:]
            else:
                raise ValueError(f"truncate_mode {self.truncate_mode} not recognized.")
        elif T < self.max_len:
            pad_shape = (self.max_len - T,) + tuple(feature.shape[1:])
            feature = torch.cat([feature, torch.zeros(pad_shape, dtype=feature.dtype)], dim=0)
            mask[T:] = 0

        if self.return_meta:
            meta = self.meta[idx]
            return images, feature, mask, label_tensor, meta
        else:
            return images, feature, mask, label_tensor

    def _sample_indices(self, num_frames):
        if num_frames <= self.num_frames:
            indices = np.linspace(0, num_frames - 1, self.num_frames, dtype=int)
        else:
            ticks = np.linspace(0, num_frames, self.num_frames + 1, dtype=int)
            indices = ticks[:-1] + (ticks[1:] - ticks[:-1]) // 2
        return indices

    def _load_image(self, directory, image_name):
        image_path = os.path.join(directory, image_name)
        image = torchvision.io.read_image(image_path)  
        
        # Resize to CLIP's expected size (224x224)
        image = torchvision.transforms.functional.resize(
            image,
            (224, 224),
            interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
            antialias=True
        )
        
        image = image.float() / 255.0
        
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(-1, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(-1, 1, 1)
        image = (image - mean) / std
        
        return image

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

    def get_action_names(self):
        """
        Returns list of action names corresponding to numerical labels
        """
        return [self.all_actions[idx] for idx in self.all_actions_numerical]

def custom_collate_fn(batch):
    images, features, masks, labels = zip(*batch)
    images = torch.stack(images)
    features = torch.stack(features)
    masks = torch.stack(masks)
    labels = torch.stack(labels)
    return images, features, masks, labels

def custom_collate_fn_meta(batch):
    images, features, masks, labels, meta = zip(*batch)
    images = torch.stack(images)
    features = torch.stack(features)
    masks = torch.stack(masks)
    labels = torch.stack(labels)
    return images, features, masks, labels, meta

class AnimalKingdomActionRecognitionDataset(Dataset):
    def __init__(self,
                 split=None,
                 data_root="data/animal_kingdom",
                 video_path="video/{video_name}.mp4",
                 meta_path="annotations/AR_metadata.xlsx",
                 return_meta=False
                 ):
        self.data_root = data_root
        self.video_path = video_path
        self.meta_path = meta_path
        self.split = split

        self.actions = [
            'Keeping still', 'Dancing', 'Doing A Neck Raise', 'Standing', 'Performing allo-grooming',
            'Swimming', 'Jumping', 'Swimming In Circles', 'Walking', 'Chirping', 'Shaking Head',
            'Sensing', 'Eating', 'Carrying', 'Performing sexual pursuit', 'Hatching', 'Escaping',
            'Leaning', 'Trapped', 'Rubbing its head', 'Disturbing Another Animal', 'Sharing Food',
            'Dancing On Water', 'Sleeping', 'Defensive Rearing', 'Detaching As A Parasite', 'Resting',
            'Falling', 'Walking On Water', 'Exiting Nest', 'Unrolling', 'Doing Push Up',
            'Doing somersault', 'Digging', 'Dead', 'Landing', 'Pecking', 'Hissing', 'Biting',
            'Urinating', 'Retaliating', 'Doing A Back Kick', 'Dying', 'Exiting Cocoon',
            'Performing copulatory mounting', 'Stinging', 'Drinking', 'Washing', 'Calling',
            'Having A Flehmen Response', 'Standing in alert', 'Climbing', 'Lying On Top',
            'Competing For Dominance', 'Lying on its side', 'Gliding', 'Coiling', 'Sinking', 'Rolling',
            'Spitting', 'Giving Birth', 'Molting', 'Fighting', 'Flapping its ears', 'Holding Hands',
            'Carrying In Mouth', 'Displaying Defensive Pose', 'Camouflaging',
            'Performing allo-preening', 'Performing sexual exploration', 'Swaying', 'Spreading',
            'Drifting', 'Turning Around', 'Wrapping Itself Around Prey', 'Hanging', 'Preying',
            'Panting', 'Doing A Backward Tilt', 'Being Eaten', 'Manipulating Object', 'Building nest',
            'Grooming', 'Spreading Wings', 'Pounding', 'Struggling', 'Immobilized', 'Startled',
            'Playing', 'Abseiling', 'Preening', 'Hugging', 'Hopping', 'Running', 'Running On Water',
            'Attending', 'Rattling', 'Pulling', 'Yawning', 'Licking', 'Unmounting', 'Retreating',
            'Swinging', 'Wrapping Prey', 'Playing Dead', 'Flapping', 'Being Carried In Mouth',
            'Flapping Tail', 'Fleeing', 'Waving', 'Performing Sexual Display', 'Showing Affection',
            'Diving', 'Defecating', 'Being Dragged', 'Exploring', 'Laying Eggs', 'Entering its nest',
            'Surfacing', 'Doing A Chin Dip', 'Barking', 'Sitting', 'Getting Bullied', 'Tail Swishing',
            'Undergoing Chrysalis', 'Doing A Face Dip', 'Chasing', 'Shaking', 'Squatting', 'Attacking',
            'Moving', 'Spitting Venom', 'Puffing its throat', 'Gasping For Air', 'Doing A Side Tilt',
            'Being Carried', 'Sleeping in its nest', 'Flying', 'Giving Off Light', 'Lying Down'
        ]

        self.return_meta = return_meta

        meta_data = pd.read_excel(os.path.join(self.data_root, self.meta_path))
        meta_data = meta_data.to_dict(orient='records')

        # Filter based on split if provided
        if split is not None:
            if split == 'train':
                meta_data = [meta for meta in meta_data if meta['type'] == 'train']
            elif split == 'test':
                meta_data = [meta for meta in meta_data if meta['type'] == 'test']
            else:
                raise ValueError(f"Split {split} not recognized. Use 'train' or 'test'")

        self.video_names = [meta['video_id'] for meta in meta_data]
        self.video_paths = [os.path.join(self.data_root, self.video_path.format(video_name=video_name))
                            for video_name in self.video_names]
        self.meta = [{'name': video_name, 'path': video_path}
                     for video_name, video_path in zip(self.video_names, self.video_paths)]

        self.annots = {}
        for meta in meta_data:
            name = meta['video_id']
            labels = read_list(meta['labels'])
            animal_actions = read_list(meta['list_animal_action'])
            self.annots[name] = (labels, animal_actions)

        actions_meta = pd.read_excel(os.path.join(self.data_root, self.meta_path), sheet_name='Action')
        actions_meta = actions_meta.to_dict(orient='records')
        self.actions_meta = {record['Action'].lower(): record for record in actions_meta}

        animals_meta = pd.read_excel(os.path.join(self.data_root, self.meta_path), sheet_name='Animal')
        animals_meta = animals_meta.to_dict(orient='records')
        self.animals_meta = {record['Animal'].lower(): record for record in animals_meta}

        self.animals_meta['common quail'] = self.animals_meta['common quail bird']
        self.animals_meta['cuckoo bird'] = self.animals_meta['common cuckoo bird']

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        video_name = self.video_names[idx]
        annot = self.annots[video_name]
        video, _, _ = torchvision.io.read_video(video_path)  # T x H x W x C

        if self.return_meta:
            meta = self.get_meta(idx)
            return video, annot, meta
        else:
            return video, annot

    def get_meta(self, idx):
        return self.meta[idx]

    def get_annot(self, idx):
        video_name = self.video_names[idx]
        return self.annots[video_name]


def read_list(list_str):
    val = ast.literal_eval(list_str)
    if isinstance(val, int):
        return (val,)
    elif isinstance(val, tuple):
        return val
    elif isinstance(val, list):
        if len(val)>0 and isinstance(val[0], str):
            return ast.literal_eval(val[0])
        else:
            return val
    else:
        raise ValueError(f"Unknown type: {val}")




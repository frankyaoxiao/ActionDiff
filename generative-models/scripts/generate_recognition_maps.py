import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from tqdm.auto import tqdm
import wandb
import numpy as np
import math
import torch.nn.functional as F
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), "../")))
from models.MSQNet import MSQNet
from models.MLP import MLP
from models.MSQNetWithoutImageCLIP import MSQNetWithoutImageCLIP, get_text_features
from models.Transformer import SimpleTransformer
from datasets.animalkingdom import AnimalKingdomFeatureDataset, custom_collate_fn, custom_collate_fn_meta

import matplotlib.pyplot as plt


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss

def parse_args():
    parser = argparse.ArgumentParser(description='Train MSQNet on AnimalKingdom dataset')
    parser.add_argument('--data_root', type=str, default='data/video_feats_new/video_feats_fp32/feats_layer_idxs=3')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--save_dir', type=str, default='results/baseline')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--use_mean', action='store_true')
    parser.add_argument('--mixup_alpha', type=float, default=0.0, help='Alpha parameter for mixup')
    parser.add_argument('--num_warmup_epochs', type=int, default=4)
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'])
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--loss_type', type=str, default='bce', choices=['bce', 'focal'])
    parser.add_argument('--max_length', type=int, default=100)
    parser.add_argument('--decoder_heads', type=int, default=8)
    parser.add_argument('--decoder_layers', type=int, default=6)
    parser.add_argument('--num_workers', type=int, default=10, help='Number of data loading workers')
    parser.add_argument('--model_type', type=str, default='msqnet', choices=['msqnet', 'mlp', 'msqnet_without_image_clip', 'transformer'])
    parser.add_argument('--ckpt_path', type=str, default=None, help='Path to checkpoint to resume training from')
    return parser.parse_args()

def mixup_data(x, y, images=None, alpha=0.2, device='cuda'):
    """Performs mixup on the input data, labels, and images if provided."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index]
    mixed_y = lam * y + (1 - lam) * y[index]
    
    mixed_images = None
    if images is not None:
        mixed_images = lam * images + (1 - lam) * images[index]
    
    return mixed_x, mixed_y, mixed_images

def train_epoch(model, train_loader, criterion, optimizer, device, mixup_alpha=0.2, scheduler=None):
    model.train()
    total_loss = 0
    
    for batch_idx, (images, features, masks, labels) in enumerate(tqdm(train_loader, desc='Training', dynamic_ncols=True)):
        images = images.to(device)
        features = features.to(device)
        masks = masks.to(device)
        labels = labels.to(device)
        
        features, labels, images = mixup_data(features, labels, images, mixup_alpha, device)
        
        optimizer.zero_grad()
        outputs = model(images, features, attention_mask=masks)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def validate(model, val_loader, criterion, dataset, device):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for images, features, masks, labels in tqdm(val_loader, desc='Validation', dynamic_ncols=True):
            images = images.to(device)
            features = features.to(device)
            masks = masks.to(device)
            labels = labels.to(device)
            
            outputs = model(images, features, attention_mask=masks)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            all_predictions.append(torch.sigmoid(outputs))
            all_targets.append(labels)
    
    avg_loss = total_loss / len(val_loader)
    
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    map_score = dataset.calculate_map(all_predictions, all_targets)
    
    return avg_loss, map_score

def get_scheduler(optimizer, num_warmup_epochs, num_training_epochs, num_steps_per_epoch):
    """Creates a learning rate scheduler with warmup and cosine decay."""
    num_warmup_steps = num_warmup_epochs * num_steps_per_epoch
    num_training_steps = num_training_epochs * num_steps_per_epoch
    
    def lr_lambda(current_step):
        # Warmup
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine decay
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def load_checkpoint(ckpt_path, model, optimizer):
    """load checkpoint and return the start epoch and best validation score."""
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
    
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_val_map = checkpoint.get('val_map', 0.0)
    
    return start_epoch, best_val_map
def main():
    args = parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    log_path = os.path.join(args.save_dir, 'training_log.txt')
    
    with open(log_path, 'w') as f:
        f.write("Training Configuration:\n")
        f.write("=====================\n")
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
        f.write("\nTraining Progress:\n")
        f.write("=================\n")
        f.write("Epoch\tTrain Loss\tVal Loss\tTrain mAP\tVal mAP\tLearning Rate\n")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
    if args.wandb:
        wandb.init(project="msqnet-animalkingdom", config=args)
    
    train_dataset = AnimalKingdomFeatureDataset(split='train', use_mean=args.use_mean, features_root=args.data_root, max_len=args.max_length)
    val_dataset = AnimalKingdomFeatureDataset(split='test', use_mean=args.use_mean, features_root=args.data_root, max_len=args.max_length, return_meta=True)
    
    train_loader = DataLoader(train_dataset, 
                             batch_size=args.batch_size, 
                             shuffle=True, 
                             num_workers=args.num_workers,
                             collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, 
                           batch_size=args.batch_size, 
                           shuffle=False, 
                           num_workers=args.num_workers,
                           collate_fn=custom_collate_fn_meta)
    
    class_names = train_dataset.all_actions

    text_embeddings = get_text_features(class_names)
    if args.model_type == 'msqnet':
        model = MSQNet(text_embeddings, args.decoder_heads, args.decoder_layers).to(args.device)
    elif args.model_type == 'mlp':
        model = MLP(1280, args.max_length, text_embeddings.shape[0]).to(args.device)
    elif args.model_type == 'msqnet_without_image_clip':
        model = MSQNetWithoutImageCLIP(text_embeddings, args.decoder_heads, args.decoder_layers).to(args.device)
    elif args.model_type == 'transformer':
        model = SimpleTransformer(text_embeddings.shape[0], args.decoder_heads, args.decoder_layers).to(args.device)
    else:
        raise ValueError(f"Invalid model type: {args.model_type}")
    model.float()
    
    if args.loss_type == 'focal':
        criterion = FocalLoss(alpha=1, gamma=2, reduction='mean')
    else:
        criterion = nn.BCEWithLogitsLoss()

    #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    num_steps_per_epoch = len(train_loader) 
    total_steps = num_steps_per_epoch * args.num_epochs
    warmup_steps = num_steps_per_epoch * args.num_warmup_epochs
    
    print(f"Total steps: {total_steps}")
    print(f"Warmup steps: {warmup_steps}")
    print(f"Steps per epoch: {num_steps_per_epoch}")
    
    if args.optimizer == 'adam':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, 
                            weight_decay=args.weight_decay)
    
    scheduler = get_scheduler(optimizer, 
                            num_warmup_epochs=args.num_warmup_epochs,
                            num_training_epochs=args.num_epochs,
                            num_steps_per_epoch=num_steps_per_epoch)
    
    start_epoch = 0
    best_val_map = 0.0
    
    if args.ckpt_path is not None:
        start_epoch, best_val_map = load_checkpoint(args.ckpt_path, model, optimizer)
        print(f"Resuming from epoch {start_epoch} with best validation mAP: {best_val_map:.4f}")
    

    model.eval()
    device = args.device

    with torch.no_grad():
        for images, features, masks, labels, meta in tqdm(val_loader, desc='Validation', dynamic_ncols=True):
            images = images.to(device)
            features = features.to(device)
            masks = masks.to(device)
            labels = labels.to(device)

            H, W = 3, 5
            action_maps = torch.zeros(args.batch_size, H, W)
            feats2d = [np.load(meta[i]['path'].replace('_mean', '')) for i in range(args.batch_size)]
            unique_labels = [torch.where(img_labels)[0][0] for img_labels in labels]
            for h in range(H):
                for w in range(W):
                    local_feats = features.clone()
                    for i in range(args.batch_size):
                        seq_len = feats2d[i].shape[0]
                        seq_len = min(seq_len, args.max_length)
                        local_feats[i, :seq_len, :] = torch.tensor(feats2d[i][:seq_len, h, w, :]).to(device)
                    outputs = model(images, local_feats, attention_mask=masks)
                    for i in range(args.batch_size):
                        action_maps[i, h, w] = outputs[i, unique_labels[i]].sigmoid()

            output_dir = "../plots/heatmaps"
            for i in range(args.batch_size):
                img = F.interpolate(images[i][8].unsqueeze(dim=0), size=(150, 250), mode="bilinear", align_corners=False).squeeze(0).permute(1,2,0).cpu().numpy()
                heatmap = F.interpolate(action_maps[i].unsqueeze(0).unsqueeze(0), size=(150, 250), mode='bilinear', align_corners=False).squeeze(0).squeeze(0).cpu().numpy()
                alpha = 0.5

                plt.figure(figsize=(5, 3))
                plt.imshow(img)
                plt.imshow(heatmap, cmap="jet", interpolation="nearest", alpha=alpha)
                # plt.colorbar()
                plt.axis("off")  # Remove axes for better visualization

                # Save the figure
                video_name = meta[i]['name']
                action_label = unique_labels[i]
                filename = os.path.join(output_dir, f"{video_name}_heatmap_action={action_label}.png")
                plt.savefig(filename, bbox_inches="tight", pad_inches=0)
                # plt.show()
                plt.close()  # Close the figure to free memory

            pass

if __name__ == '__main__':
    main()


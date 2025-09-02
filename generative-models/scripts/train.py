import os
import sys
sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(__file__), "../")))
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from tqdm.auto import tqdm
import numpy as np
import math
import torch.nn.functional as F
from models.MSQNet import MSQNet
from models.MLP import MLP
from models.MSQNetWithoutImageCLIP import MSQNetWithoutImageCLIP, get_text_features
from models.Transformer import ClassificationTransformer
from datasets.animalkingdom import AnimalKingdomFeatureDataset, AnimalKingdomNewSpeciesDataset, custom_collate_fn, AnimalKingdomNewActionsDataset
from datasets.charades import CharadesFeatureDataset, CharadesFeatureDatasetSplit
from datasets.ucf import UCFFeatureDataset
from datasets.hmdb import HMDBFeatureDataset
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data/video_feats_new/video_feats_fp32/feats_layer_idxs=3')
    parser.add_argument('--data_root2', type=str, default=None)
    parser.add_argument('--trainviewpoint', type=int, default=None)
    parser.add_argument('--testviewpoint', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--save_dir', type=str, default='results/baseline')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--use_mean', action='store_true')
    parser.add_argument('--use_all', action='store_true')
    parser.add_argument('--mixup_alpha', type=float, default=0.0, help='Alpha parameter for mixup')
    parser.add_argument('--num_warmup_epochs', type=int, default=4)
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'])
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--loss_type', type=str, default='bce', choices=['bce', 'focal', 'mse', 'ce'])
    parser.add_argument('--max_length', type=int, default=100)
    parser.add_argument('--decoder_heads', type=int, default=8)
    parser.add_argument('--decoder_layers', type=int, default=6)
    parser.add_argument('--num_workers', type=int, default=10, help='Number of data loading workers')
    parser.add_argument('--model_type', type=str, default='msqnet', choices=['msqnet', 'mlp', 'msqnet_without_image_clip', 'classificationtransformer'])
    parser.add_argument('--ckpt_path', type=str, default=None, help='Path to checkpoint to resume training from')
    parser.add_argument('--dataset', type=str, default='animalkingdom', choices=['animalkingdom', 'charades', 'animalkingdom-new-actions', 'animalkingdom-new-species', 'ucf-hmdb', 'hmdb-ucf', 'ucf', 'hmdb'],
                       help='Dataset to use for training')
    parser.add_argument('--focal_alpha', type=float, default=1.0, help='Alpha parameter for focal loss')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='Gamma parameter for focal loss')
    parser.add_argument('--split_actions', action='store_true', help='Split videos into multiple actions on charades')
    parser.add_argument('--sched_type', type=str, default='cosineannealing', choices=['cosineannealing', 'cosineannealing_wr'],
                        help='Type of learning rate scheduler: standard cosine annealing or cosine annealing with warm restarts')
    parser.add_argument('--T_0', type=float, default=3, help='Initial restart period for CosineAnnealingWarmRestarts scheduler (in epochs)')
    parser.add_argument('--T_mult', type=float, default=1, help='Multiplier for restart period after every restart for CosineAnnealingWarmRestarts scheduler')
    parser.add_argument('--truncate_mode', type=str, default='beg', choices=['beg', 'mid', 'end'], 
                        help='How to truncate sequences that exceed max_length: beginning, middle, or end')
    parser.add_argument('--num_validations_per_epoch', type=int, default=1, help='Number of validation runs within each epoch')
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

def train_epoch(model, train_loader, criterion, optimizer, device, mixup_alpha=0.2, scheduler=None, sched_type='cosineannealing', current_epoch=0, num_validations=1, val_loader=None, val_dataset=None):
    model.train()
    total_loss = 0
    num_batches = len(train_loader)
    best_val_map_in_epoch = -float('inf')
    if num_validations > 1 and val_loader is not None:
        val_interval = max(1, num_batches // num_validations)
    else:
        val_interval = None
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
            if sched_type == 'cosineannealing_wr':
                scheduler.step(current_epoch + (batch_idx + 1) / num_batches)
            else:
                scheduler.step()
        
        total_loss += loss.item()
        
        if val_interval is not None and ((batch_idx+1) % val_interval == 0 or (batch_idx+1) == num_batches):
            model.eval()
            interim_val_loss, interim_val_map = validate(model, val_loader, criterion, val_dataset, device)
            print(f"Intermediate validation at batch {batch_idx+1}/{num_batches} - Val Loss: {interim_val_loss:.4f}, Val mAP: {interim_val_map:.4f}\n")
            best_val_map_in_epoch = max(best_val_map_in_epoch, interim_val_map)
            model.train()
    
    avg_loss = total_loss / num_batches
    if val_interval is not None:
        return avg_loss, best_val_map_in_epoch
    else:
        return avg_loss

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
    start_epoch = checkpoint['epoch'] 
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
        f.write("Epoch\tTrain Loss\tVal Loss\tVal mAP\tLearning Rate\n")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    
    if args.dataset == 'animalkingdom':
        train_dataset = AnimalKingdomFeatureDataset(
            split='train', 
            use_mean=args.use_mean, 
            use_all=args.use_all,
            features_root=args.data_root, 
            max_len=args.max_length,
            truncate_mode=args.truncate_mode
        )
        val_dataset = AnimalKingdomFeatureDataset(
            split='test', 
            use_mean=args.use_mean, 
            use_all=args.use_all,
            features_root=args.data_root, 
            max_len=args.max_length,
            truncate_mode=args.truncate_mode
        )
    elif args.dataset == 'animalkingdom-new-species':
        train_dataset = AnimalKingdomNewSpeciesDataset(
            split='train',
            features_root=args.data_root,
            use_mean=args.use_mean,
            max_len=args.max_length
        )
        val_dataset = AnimalKingdomNewSpeciesDataset(
            split='test',
            features_root=args.data_root,
            use_mean=args.use_mean,
            max_len=args.max_length
        )
    elif args.dataset == 'animalkingdom-new-actions':
        train_dataset = AnimalKingdomNewActionsDataset(
            split='train',
            features_root=args.data_root,
            use_mean=args.use_mean,
            max_len=args.max_length
        )
        val_dataset = AnimalKingdomNewActionsDataset(
            split='test',
            features_root=args.data_root,
            use_mean=args.use_mean,
            max_len=args.max_length
        )
    elif args.dataset == 'ucf-hmdb':
        train_dataset = UCFFeatureDataset(
            split='train',
            features_root=args.data_root,
            use_mean=args.use_mean,
            max_len=args.max_length,
        )
        val_dataset = UCFFeatureDataset(
            split='val',
            features_root=args.data_root,
            use_mean=args.use_mean,
            max_len=args.max_length,
        )
        test_dataset = HMDBFeatureDataset(
            split='full',
            features_root=args.data_root2,
            use_mean=args.use_mean,
            max_len=args.max_length,
        )
    elif args.dataset == 'hmdb-ucf':
        train_dataset = HMDBFeatureDataset(
            split='train',
            features_root=args.data_root,
            use_mean=args.use_mean,
            max_len=args.max_length,
        )
        val_dataset = HMDBFeatureDataset(
            split='val',
            features_root=args.data_root,
            use_mean=args.use_mean,
            max_len=args.max_length,
        )
        test_dataset = UCFFeatureDataset(
            split='full',
            features_root=args.data_root2,
            use_mean=args.use_mean,
        )
    elif args.dataset == 'ucf':
        train_dataset = UCFFeatureDataset(
            split='train',
            features_root=args.data_root,
            use_mean=args.use_mean,
            max_len=args.max_length,
        )
        val_dataset = UCFFeatureDataset(
            split='val',
            features_root=args.data_root,
            use_mean=args.use_mean,
            max_len=args.max_length,
        )
    elif args.dataset == 'hmdb':
        train_dataset = HMDBFeatureDataset(
            split='train',
            features_root=args.data_root,
            use_mean=args.use_mean,
            max_len=args.max_length,
        )
        val_dataset = HMDBFeatureDataset(
            split='val',
            features_root=args.data_root,
            use_mean=args.use_mean,
            max_len=args.max_length,
        )
    else:  # charades
        if args.split_actions:
            train_dataset = CharadesFeatureDatasetSplit(
                split='train',
                features_root=args.data_root,
                max_len=300,
                viewpoint=3,
                use_mean=args.use_mean
            )
        else:
            train_dataset = CharadesFeatureDataset(
                split='train',
                features_root=args.data_root,
                max_len=args.max_length,
                viewpoint=args.trainviewpoint,
                use_mean=args.use_mean
            )
        val_dataset = CharadesFeatureDataset(
            split='test',
            features_root=args.data_root,
            max_len=args.max_length,
            viewpoint=args.testviewpoint,
            use_mean=args.use_mean
        )
    
    train_loader = DataLoader(train_dataset, 
                             batch_size=args.batch_size, 
                             shuffle=True, 
                             num_workers=args.num_workers,
                             collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, 
                           batch_size=args.batch_size, 
                           shuffle=False, 
                           num_workers=args.num_workers,
                           collate_fn=custom_collate_fn)
    if args.dataset == 'ucf-hmdb' or args.dataset == 'hmdb-ucf':
        test_loader = DataLoader(test_dataset, 
                           batch_size=args.batch_size, 
                           shuffle=False, 
                           num_workers=args.num_workers,
                           collate_fn=custom_collate_fn)
    else:
        test_loader = None
    class_names = train_dataset.all_actions

    text_embeddings = get_text_features(class_names)
    if args.model_type == 'msqnet':
        model = MSQNet(text_embeddings, args.decoder_heads, args.decoder_layers).to(args.device)
    elif args.model_type == 'mlp':
        model = MLP(train_dataset.embed_dim, args.max_length, text_embeddings.shape[0]).to(args.device)
    elif args.model_type == 'msqnet_without_image_clip':
        model = MSQNetWithoutImageCLIP(text_embeddings, args.decoder_heads, args.decoder_layers).to(args.device)
    elif args.model_type == 'transformer':
        model = SimpleTransformer(text_embeddings.shape[0], train_dataset.embed_dim, args.decoder_heads, args.decoder_layers).to(args.device)
    elif args.model_type == 'fulltransformer':
        model = FullTransformer(text_embeddings, args.decoder_heads, args.decoder_layers).to(args.device)
    elif args.model_type == 'classificationtransformer':
        model = ClassificationTransformer(text_embeddings.shape[0], train_dataset.embed_dim, args.decoder_heads, args.decoder_layers).to(args.device)
    else:
        raise ValueError(f"Invalid model type: {args.model_type}")
    model.float()
    
    if args.loss_type == 'focal':
        criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma, reduction='mean')
    elif args.loss_type == 'mse':
        criterion = nn.MSELoss()
    elif args.loss_type == 'ce':
        criterion = nn.CrossEntropyLoss()
    else:  # bce
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
    
    # Select scheduler based on parser parameter
    if args.sched_type == 'cosineannealing_wr':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                        optimizer, T_0=args.T_0, T_mult=args.T_mult, eta_min=0)
    else:
        scheduler = get_scheduler(optimizer, 
                                  num_warmup_epochs=args.num_warmup_epochs,
                                  num_training_epochs=args.num_epochs,
                                  num_steps_per_epoch=num_steps_per_epoch)
    
    start_epoch = 0
    best_val_map = 0.0
    
    if args.ckpt_path is not None:
        start_epoch, best_val_map = load_checkpoint(args.ckpt_path, model, optimizer)
        print(f"Resuming from epoch {start_epoch} with best validation mAP: {best_val_map:.4f}")
    
    for epoch in range(start_epoch, args.num_epochs):
        print(f"\nEpoch {epoch+1}/{args.num_epochs}")
        if args.num_validations_per_epoch > 1:
            train_loss, epoch_val_map = train_epoch(
                model, train_loader, criterion, optimizer, args.device, args.mixup_alpha,
                scheduler=scheduler, sched_type=args.sched_type, current_epoch=epoch,
                num_validations=args.num_validations_per_epoch, val_loader=val_loader, val_dataset=val_dataset
            )
            val_map = epoch_val_map
            val_loss = -1.0
        else:
            train_loss = train_epoch(
                model, train_loader, criterion, optimizer, args.device, args.mixup_alpha,
                scheduler=scheduler, sched_type=args.sched_type, current_epoch=epoch,
                num_validations=args.num_validations_per_epoch
            )
            val_loss, val_map = validate(model, val_loader, criterion, val_dataset, args.device)
            if args.dataset == 'epic-kitchens' or args.dataset == 'ucf-hmdb' or args.dataset == 'hmdb-ucf':
                test_loss, test_map = validate(model, test_loader, criterion, test_dataset, args.device)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning Rate: {current_lr:.6f}")
        
        print(f"Train Loss: {train_loss:.4f}")
        if val_loss >= 0:
            print(f"Val Loss: {val_loss:.4f}")
        print(f"Val mAP: {val_map:.4f}")
        if args.dataset == 'epic-kitchens' or args.dataset == 'ucf-hmdb' or args.dataset == 'hmdb-ucf':
            print(f"Test Loss: {test_loss:.4f}")
            print(f"Test mAP: {test_map:.4f}")
        
        with open(log_path, 'a') as f:
            val_loss_str = f"{val_loss:.4f}" if val_loss >= 0 else "NA"
            f.write(f"{epoch+1}\t{train_loss:.4f}\t{val_loss_str}\t{val_map:.4f}\t{current_lr:.6f}\n")
            if args.dataset == 'epic-kitchens' or args.dataset == 'ucf-hmdb' or args.dataset == 'hmdb-ucf':
                test_loss_str = f"{test_loss:.4f}" if test_loss >= 0 else "NA"
                f.write(f"{epoch+1} (test)\t{train_loss:.4f}\t{test_loss_str}\t{test_map:.4f}\t{current_lr:.6f}\n")
        
        if val_map > best_val_map:
            best_val_map = val_map
            #torch.save({
            #    'epoch': epoch,
            #    'model_state_dict': model.state_dict(),
            #    'optimizer_state_dict': optimizer.state_dict(),
            #    'val_map': val_map,
            #}, os.path.join(args.save_dir, 'best_model.pth'))
        
        #torch.save({
        #    'epoch': epoch,
        #    'model_state_dict': model.state_dict(),
        #    'optimizer_state_dict': optimizer.state_dict(),
        #    'val_map': val_map,
        #}, os.path.join(args.save_dir, 'latest_model.pth'))

    print(f"\nTraining completed. Best validation mAP: {best_val_map:.4f}")
    
    with open(log_path, 'a') as f:
        f.write("\nTraining Summary:\n")
        f.write("================\n")
        f.write(f"Best validation mAP: {best_val_map:.4f}\n")

if __name__ == '__main__':
    main()


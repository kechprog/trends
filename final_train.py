import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import numpy as np
import json
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from model import MarketRegimeTransformer


class MarketRegimeDataset(Dataset):
    def __init__(self, X_path, y_class_path, y_duration_path):
        self.X = np.load(X_path).astype(np.float32)
        self.y_class = np.load(y_class_path).astype(np.long)
        self.y_duration = np.load(y_duration_path).astype(np.float32)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y_class[idx], self.y_duration[idx]


def create_data_loaders(data_dir, batch_size=64, val_split=0.06, num_workers=4):
    X_path = os.path.join(data_dir, 'X_train_n128.npy')
    y_class_path = os.path.join(data_dir, 'y_class_n128.npy')
    y_duration_path = os.path.join(data_dir, 'y_duration_n128.npy')
    
    dataset = MarketRegimeDataset(X_path, y_class_path, y_duration_path)
    
    indices = list(range(len(dataset)))
    train_indices, val_indices = train_test_split(indices, test_size=val_split, random_state=42)
    
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    
    train_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, len(train_indices), len(val_indices)


def train_epoch(model, train_loader, criterion_class, criterion_duration, optimizer, device, class_weight=1.0, duration_weight=0.1):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target_class, target_duration) in enumerate(train_loader):
        data = data.to(device)
        target_class = target_class.to(device)
        target_duration = target_duration.to(device).unsqueeze(1)
        
        optimizer.zero_grad()
        
        class_logits, duration_pred = model(data)
        
        loss_class = criterion_class(class_logits, target_class)
        loss_duration = criterion_duration(duration_pred, target_duration)
        
        loss = class_weight * loss_class + duration_weight * loss_duration
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = class_logits.max(1)
        total += target_class.size(0)
        correct += predicted.eq(target_class).sum().item()
        
        if batch_idx % 100 == 0:
            print(f'  Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f} '
                  f'(Class: {loss_class.item():.4f}, Duration: {loss_duration.item():.4f})')
    
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(train_loader)
    
    return avg_loss, accuracy


def validate(model, val_loader, criterion_class, criterion_duration, device, class_weight=1.0, duration_weight=0.1):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    class_correct = {0: 0, 1: 0, 2: 0}
    class_total = {0: 0, 1: 0, 2: 0}
    duration_mse = 0
    
    with torch.no_grad():
        for data, target_class, target_duration in val_loader:
            data = data.to(device)
            target_class = target_class.to(device)
            target_duration = target_duration.to(device).unsqueeze(1)
            
            class_logits, duration_pred = model(data)
            
            loss_class = criterion_class(class_logits, target_class)
            loss_duration = criterion_duration(duration_pred, target_duration)
            loss = class_weight * loss_class + duration_weight * loss_duration
            
            total_loss += loss.item()
            _, predicted = class_logits.max(1)
            total += target_class.size(0)
            correct += predicted.eq(target_class).sum().item()
            
            # Track duration MSE
            duration_mse += ((duration_pred - target_duration) ** 2).sum().item()
            
            for i in range(target_class.size(0)):
                label = target_class[i].item()
                class_total[label] += 1
                if predicted[i] == label:
                    class_correct[label] += 1
    
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(val_loader)
    avg_duration_mse = duration_mse / total
    
    class_accuracies = {
        'bull': 100. * class_correct[0] / class_total[0] if class_total[0] > 0 else 0,
        'flat': 100. * class_correct[1] / class_total[1] if class_total[1] > 0 else 0,
        'bear': 100. * class_correct[2] / class_total[2] if class_total[2] > 0 else 0
    }
    
    return avg_loss, accuracy, class_accuracies, avg_duration_mse


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory Usage:")
        print(f"  Allocated: {round(torch.cuda.memory_allocated(0)/1024**3,1)} GB")
        print(f"  Cached: {round(torch.cuda.memory_reserved(0)/1024**3,1)} GB")
        print(f"  Total VRAM: {round(torch.cuda.get_device_properties(0).total_memory/1024**3,1)} GB")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    data_dir = 'data/training'
    batch_size = 320
    num_epochs = 80
    learning_rate = 1e-4
    class_weight = 1.0
    
    # Duration weight will gradually increase from 0.05 to 0.4
    duration_weight_start = 0.05
    duration_weight_end = 0.4
    
    print("Loading data...")
    train_loader, val_loader, train_size, val_size = create_data_loaders(
        data_dir, 
        batch_size=batch_size,
        val_split=0.06,
        num_workers=8
    )
    print(f"Training samples: {train_size}, Validation samples: {val_size}")
    
    # Calculate class weights based on the distribution
    # From summary: bull: 43.9%, flat: 25.8%, bear: 30.3%
    class_weights = torch.tensor([1.0/0.439, 1.0/0.258, 1.0/0.303]).to(device)
    class_weights = class_weights / class_weights.mean()  # Normalize
    print(f"Class weights: Bull={class_weights[0]:.3f}, Flat={class_weights[1]:.3f}, Bear={class_weights[2]:.3f}")
    
    model = MarketRegimeTransformer(
        lookback_window=127,
        n_features=5,
        d_model=256,
        nhead=8,
        num_feature_layers=4,
        num_aggregate_layers=3,
        dim_feedforward=1024,
        dropout=0.1,
        num_classes=3
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Use weighted cross entropy loss
    criterion_class = nn.CrossEntropyLoss(weight=class_weights)
    criterion_duration = nn.MSELoss()
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Two-phase scheduler: warmup then cosine decay
    warmup_epochs = 10
    scheduler1 = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs-warmup_epochs)
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[warmup_epochs])
    
    best_val_accuracy = 0
    best_balanced_accuracy = 0
    best_duration_mse = float('inf')
    best_epoch = 0
    
    os.makedirs('checkpoints', exist_ok=True)
    
    for epoch in range(num_epochs):
        # Gradually increase duration weight
        progress = epoch / num_epochs
        duration_weight = duration_weight_start + (duration_weight_end - duration_weight_start) * progress
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Duration weight: {duration_weight:.3f}")
        print("-" * 50)
        
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion_class, criterion_duration, 
            optimizer, device, class_weight, duration_weight
        )
        
        val_loss, val_acc, class_accuracies, val_duration_mse = validate(
            model, val_loader, criterion_class, criterion_duration, 
            device, class_weight, duration_weight
        )
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\nTrain Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Val Duration MSE: {val_duration_mse:.4f} (RMSE: {np.sqrt(val_duration_mse):.4f})")
        print(f"Class Accuracies - Bull: {class_accuracies['bull']:.2f}%, "
              f"Flat: {class_accuracies['flat']:.2f}%, Bear: {class_accuracies['bear']:.2f}%")
        print(f"Learning Rate: {current_lr:.2e}")
        
        if device.type == 'cuda':
            print(f"GPU Memory: {round(torch.cuda.memory_allocated(0)/1024**3,2)} GB allocated")
        
        # Calculate balanced accuracy
        balanced_accuracy = (class_accuracies['bull'] + class_accuracies['flat'] + class_accuracies['bear']) / 3
        
        # Save best model based on multiple criteria
        save_model = False
        save_reason = ""
        
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            save_model = True
            save_reason = "best overall accuracy"
        
        if balanced_accuracy > best_balanced_accuracy and val_duration_mse < 15:
            best_balanced_accuracy = balanced_accuracy
            save_model = True
            save_reason = "best balanced accuracy"
        
        if val_duration_mse < best_duration_mse and val_acc > 65:
            best_duration_mse = val_duration_mse
            save_model = True
            save_reason = "best duration MSE"
        
        if save_model:
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'val_duration_mse': val_duration_mse,
                'class_accuracies': class_accuracies,
                'balanced_accuracy': balanced_accuracy,
                'duration_weight': duration_weight
            }, 'checkpoints/best_model_final.pth')
            print(f"  -> New best model saved! ({save_reason})")
    
    print(f"\nTraining complete!")
    print(f"Best validation accuracy: {best_val_accuracy:.2f}%")
    print(f"Best balanced accuracy: {best_balanced_accuracy:.2f}%")
    print(f"Best duration MSE: {best_duration_mse:.4f} (RMSE: {np.sqrt(best_duration_mse):.4f})")
    print(f"Best model saved at epoch {best_epoch}")


if __name__ == "__main__":
    main()
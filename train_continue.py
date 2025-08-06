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
    
    # Load checkpoint
    checkpoint_path = 'checkpoints/best_model.pth'
    print(f"\nLoading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    data_dir = 'data/training'
    batch_size = 320  # Increased 2.5x from 128
    num_epochs = 30  # Continue for 30 more epochs
    learning_rate = 5e-5  # Lower learning rate for fine-tuning
    class_weight = 1.0
    duration_weight = 0.35  # Increased from 0.1
    
    print("Loading data...")
    train_loader, val_loader, train_size, val_size = create_data_loaders(
        data_dir, 
        batch_size=batch_size,
        val_split=0.06,
        num_workers=8
    )
    print(f"Training samples: {train_size}, Validation samples: {val_size}")
    
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
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Loaded model from epoch {checkpoint['epoch'] + 1}")
    print(f"Previous best val accuracy: {checkpoint['val_acc']:.2f}%")
    print(f"Previous class loss: {checkpoint['val_loss']:.4f}")
    
    criterion_class = nn.CrossEntropyLoss()
    criterion_duration = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Don't load optimizer state to start fresh with new learning rate
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    # Use a constant LR with manual decay
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    best_val_accuracy = checkpoint['val_acc']
    best_epoch = checkpoint['epoch'] + 1
    start_epoch = checkpoint['epoch'] + 1
    
    print(f"\nStarting training from epoch {start_epoch + 1}")
    print(f"New loss weights - Class: {class_weight}, Duration: {duration_weight}")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {start_epoch + epoch + 1}/{start_epoch + num_epochs}")
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
        
        print(f"\nTrain Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Val Duration MSE: {val_duration_mse:.4f} (RMSE: {np.sqrt(val_duration_mse):.4f})")
        print(f"Class Accuracies - Bull: {class_accuracies['bull']:.2f}%, "
              f"Flat: {class_accuracies['flat']:.2f}%, Bear: {class_accuracies['bear']:.2f}%")
        
        if device.type == 'cuda':
            print(f"GPU Memory: {round(torch.cuda.memory_allocated(0)/1024**3,2)} GB allocated")
        
        # Save if validation accuracy improved OR duration MSE improved significantly
        if val_acc > best_val_accuracy or (val_acc >= best_val_accuracy - 0.5 and val_duration_mse < 40):
            best_val_accuracy = val_acc
            best_epoch = start_epoch + epoch + 1
            torch.save({
                'epoch': start_epoch + epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'val_duration_mse': val_duration_mse,
                'class_accuracies': class_accuracies
            }, 'checkpoints/best_model_duration_focus.pth')
            print(f"  -> New best model saved! (Duration MSE: {val_duration_mse:.4f})")
    
    print(f"\nTraining complete! Best validation accuracy: {best_val_accuracy:.2f}% at epoch {best_epoch}")


if __name__ == "__main__":
    main()
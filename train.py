"""Training script."""

import argparse
import torch
import torch.optim as optim

from config import get_config
from model import create_model
from data import get_dataloader
from loss import get_loss_fn
from metrics import compute_metrics
from utils import set_seed, get_device
from evaluate import evaluate
from tqdm import tqdm

DEFAULT_CONFIG = 'baseline'

def train_epoch (model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()

    all_losses = []
    accuracies = []

    correct_counter = 0
    sample_counter = 0

    pbar = tqdm(dataloader,desc="Training")

    for X,y in pbar:
        X,y = X.to(device), y.to(device)

        preds = model(X)
        loss = criterion(preds, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        metrics = compute_metrics(preds, y)

        correct_counter += metrics * y.size(0)
        sample_counter += y.size(0)

        all_losses.append(loss.item())
        accuracies.append(metrics)
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "accuracy": f"{metrics * 100:.2f}%"})
    avg_loss = sum(all_losses) / len(all_losses)
    accuracy = (correct_counter / sample_counter) * 100
    print(f"Train Loss: {avg_loss:.4f} | Train Accuracy: {accuracy:.2f}%")

    return {
        'losses': all_losses,
        'accuracies': accuracies
    }


def train(config):
    """Main training function."""
    set_seed(config.seed)
    device = get_device(config.device)

    history = {'train_loss': [],
             'train_accuracy': [],
             'valid_loss': [],
             'valid_accuracy': [],
             'valid_error': []}
    
    best_val_error = 100
    
    # Setup
    train_loader = get_dataloader(config, 'train')
    val_loader = get_dataloader(config, 'val')
    model = create_model(config).to(device)
    criterion = get_loss_fn()

    if (config.optimizer_type == 'sgd'):
        optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum= config.momentum)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size= 3, gamma= 0.5)
    else:
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate,weight_decay= config.decay)
        scheduler = None

     

    # Training loop
    for epoch in range(config.num_epochs):
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        if scheduler: 
           scheduler.step()
        
        history['train_loss'].append(train_metrics['losses'])
        history['train_accuracy'].append(train_metrics['accuracies'])
        history['valid_loss'].append(val_metrics['losses'])
        history['valid_accuracy'].append(val_metrics['accuracies'])
        history['valid_error'].append(val_metrics['error'])

        current_lr = optimizer.param_groups[0]['lr']
        print(f"current_lr : {current_lr:.6f} | epoch: {epoch+1}/{config.num_epochs}")

        torch.save(model.state_dict(), f"{config.name}_last.pth")

        if val_metrics['error'] < best_val_error:
            best_val_error = val_metrics['error']
            torch.save(model.state_dict(), f"{config.name}_best.pth")
            print(f"New best model saved! Erorr : {val_metrics['error']:.2f}%")


    print(f"best_val_error {best_val_error} __ paper error :15.65")
    return history
       
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG,
                       choices=['baseline', 'ablation1', 'ablation2', 'ablation3'],
                       help='Configuration to use')
    args = parser.parse_args()
    
    config = get_config(args.config)
    train(config)


if __name__ == "__main__":
    main()

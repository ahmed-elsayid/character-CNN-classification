"""Evaluation script."""

import argparse
import torch

from config import get_config
from model import create_model
from data import get_dataloader
from loss import get_loss_fn
from metrics import compute_metrics
from utils import set_seed, get_device
from tqdm import tqdm


def evaluate(model, dataloader, criterion, device):
    """Evaluate model."""
    model.eval()

    correct_counter = 0
    sample_counter = 0

    all_losses = []
    accuracies = []

    pbar = tqdm(dataloader ,desc= "Validation")

    with torch.no_grad():
        for X,y in pbar:

            X,y = X.to(device), y.to(device)

            preds = model(X)

            loss = criterion(preds, y)
            metrics = compute_metrics(preds, y)

            correct_counter += metrics * y.size(0)
            sample_counter += y.size(0)

            all_losses.append(loss.item())
            accuracies.append(metrics)

    avg_loss = sum(all_losses) / len(all_losses)
    accuracy = (correct_counter / sample_counter) * 100
    error = 100 - accuracy

    print(f'loss : {avg_loss:.4f}, acuuracy : {accuracy:.4f} error : {error:.4f}')

    return {
        'losses': all_losses,
        'accuracies': accuracies,
        'error' : error
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    args = parser.parse_args()
    
    config = get_config(args.config)
    set_seed(config.seed)
    device = get_device(config.device)
    
    test_loader = get_dataloader(config, 'test')
    model = create_model(config).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    criterion = get_loss_fn()
    results = evaluate(model, test_loader, criterion, device)

    print(f"avg Test Loss: {sum(results['losses']) / len(results['loss']) : .4f} %")
    print(f"avg Test Accuracy: {sum(results['accuracies']) / len(results['accuracies']) :.4f} %")
    print(f"Test Error: {sum(results['error']) :.4f} % paper baseline error : 15.65 %")



if __name__ == "__main__":
    main()

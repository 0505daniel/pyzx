import wandb
import torch
from torch.utils.data import random_split
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch_geometric.loader import DataLoader

from net import ZXGNN
from utils_train import ZXGraphDataset
from utils_train import get_loss_function, get_optimizer

def mape(pred, target):
    return torch.mean(torch.abs((target - pred) / target)) * 100

def train(config, root="./data", name="zx", num_samples=12800, train_ratio=0.8):
    run = wandb.init(project='zx-nco', config=config)
    config = wandb.config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = ZXGraphDataset(root, name=name, num_samples=num_samples)

    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    test_size = dataset_size - train_size

    # Optional: fix the random seed for reproducibility
    train_dataset, test_dataset = random_split(
        dataset, 
        [train_size, test_size],
        generator=torch.Generator().manual_seed(55555)
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        pin_memory=True, 
        num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        pin_memory=True, 
        num_workers=4
    )

    model = ZXGNN(
        encoding_dim=config.encoding_dim,
        heads=config.heads,
        beta=config.beta,
        dropout=config.dropout,
        normalization=config.normalization,
        num_layers=config.num_layers,
        activation=config.activation,
        readout=config.readout
    ).to(device)
    
    optimizer = get_optimizer(config.optimizer, model, config.initial_lr)
    criterion = get_loss_function(config.loss_function)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=config.T_0)

    wandb.watch(model, log="all", log_freq=1)

    best_val_mape = float('inf')
    
    for epoch in range(config.epochs):
        total_loss = 0
        total_mape = 0
        for data in train_loader:
            data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, data.y.view(-1, 1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_mape += mape(out, data.y.view(-1, 1)).item()

        avg_train_loss = total_loss / len(train_loader)
        avg_train_mape = total_mape / len(train_loader)
        wandb.log({"epoch": epoch, "train_loss": avg_train_mape})

        model.eval() # Switch to eval mode
        with torch.no_grad():
            val_loss = 0
            val_mape = 0
            for data in test_loader:
                data.to(device)
                out = model(data)
                loss = criterion(out, data.y.view(-1, 1))
                val_loss += loss.item()
                val_mape += mape(out, data.y.view(-1, 1)).item()
            avg_val_loss = val_loss / len(test_loader)
            avg_val_mape = val_mape / len(test_loader)
            wandb.log({"epoch": epoch, "val_loss": avg_val_mape})

            if avg_val_mape < best_val_mape:
                best_val_mape = avg_val_mape
                torch.save(model.state_dict(), f"trained/{name}_{num_samples}_{epoch}.pth")
        
        model.train() # Switch back to train mode
        scheduler.step()

        print(f'Epoch {epoch}, Train Loss: {avg_train_loss}, Train MAPE : {avg_train_mape}, Val Loss: {avg_val_loss}, Val MAPE: {avg_val_mape}')

    run.finish()


if __name__ == "__main__":
    config = {
    'encoding_dim': 8,
    'batch_size': 16,
    'heads': 4,
    'dropout': 0,
    'initial_lr': 0.01,
    'normalization': 'graph_norm',
    'optimizer': 'adam',
    'activation': 'elu',
    'num_layers': 4,
    'loss_function': 'mse',
    'epochs': 100, 
    'T_0': 128,
    'beta': False,
    'readout': 'attention'
}
    train(config, num_samples=100)
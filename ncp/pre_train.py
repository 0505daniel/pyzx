import wandb
import torch
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from net import ZXGNN
from utils_train import ZXGraphDataset
from utils_train import get_loss_function, get_optimizer

sweep_config = {
    'method': 'bayes',
    'metric': {
        'name': 'val_loss',
        'goal': 'minimize'
    },
    'parameters': {
        'encoding_dim': {
            'values': [4, 8, 16, 32]
        },
        'batch_size': {
            'values': [16, 32, 64, 128, 256]
        },
        'dropout': {
            'values': [0.0, 0.2, 0.5]
        },
        'initial_lr': {
            'values': [0.01, 0.001, 0.005, 0.0001]
        },
        'heads': {
            'values': [1, 2, 4]
        },
        'normalization': {
            'values': ['batch_norm', 'layer_norm', 'instance_norm', 'graph_norm']
        },
        'optimizer': {
            'values': ['adam', 'sgd', 'rmsprop', 'adagrad']
        },
        'activation': {
            'values': ['elu']
        },
        'num_layers': {
            'values': [2, 3, 4, 5, 6]
        },
        'loss_function': {
            'values': ['mse', 'huber', 'smooth_l1']
        },
        'T_0':{
            'values': [16, 32, 64, 128]
        },
        'beta': {
            'values': [True, False]
        },
        'readout':{
            'values': ['mean', 'max', 'sum', 'attention']
        }
    }
}
sweep_id = wandb.sweep(sweep_config, project="zx-nco")

def mape(pred, target):
    return torch.mean(torch.abs((target - pred) / target)) * 100

def train(config, root=".", name=None, num_samples=12800, train_ratio=0.8):
    run = wandb.init(project='zx-nco', config=config)
    config = wandb.config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = ZXGraphDataset(root, name=name, num_samples=num_samples)

    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    test_size = dataset_size - train_size

    # Optional: fix the random seed for reproducibility
    train_dataset, val_dataset = random_split(
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
    val_loader = DataLoader(
        val_dataset, 
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
        readout=config.readout_type
    ).to(device)
    
    optimizer = get_optimizer(config.optimizer, model, config.initial_lr)
    criterion = get_loss_function(config.loss_function)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=config.T_0)

    wandb.watch(model, log="all", log_freq=1)

    best_val_loss = float('inf')
    # patience = 10  # Number of epochs to wait before early stopping
    # no_improvement_epochs = 0
    
    for epoch in range(100):
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

        model.eval()
        with torch.no_grad():
            val_loss = 0
            val_mape = 0
            for data in val_loader:
                data.to(device)
                out = model(data)
                loss = criterion(out, data.y.view(-1, 1))
                val_loss += loss.item()
                val_mape += mape(out, data.y.view(-1, 1)).item()
            avg_val_loss = val_loss / len(val_loader)
            avg_val_mape = val_mape / len(val_loader)
            wandb.log({"epoch": epoch, "val_loss": avg_val_mape})

            # Check for improvement
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                # no_improvement_epochs = 0
                # Save the best model if desired
                # torch.save(model.state_dict(), f"best_model_sweep_{run.id}_{epoch}.pth")
            # else:
            #     no_improvement_epochs += 1
            #     if no_improvement_epochs >= patience:
            #         print(f"Early stopping at epoch {epoch}. No improvement in validation loss for {patience} epochs.")
            #         break
        
        model.train()
        scheduler.step()

        print(f'Epoch {epoch}, Train Loss: {avg_train_loss}, Train MAPE: {avg_train_mape}, Val Loss: {avg_val_loss}, Val MAPE: {avg_val_mape}')

    run.finish()

if __name__ == "__main__":
    wandb.agent(sweep_id, train, count=100)
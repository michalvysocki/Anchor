import numpy as np
import torch
from config import (
    BURN_IN,
    DEVICE,
    HIDDEN_SIZE,
    SEED,
    TRAIN_DATA_SEED,
    TRAINING_SEEDS,
    VAL_DATA_SEED,
    ObservabilityMode,
    model_path,
)
from model import Predictor
from torch import nn
from torch.optim import Adam
from toystateworld import generate_dataset, observation_dim


def training(
    mode: ObservabilityMode,
    n_traj: int = 1000,
    len_traj: int = 36,
    burn: int = BURN_IN,
    hidden_size: int = HIDDEN_SIZE,
    n_epochs: int = 10000,
    learning_rate: float = 1e-3,
    seed: int = SEED,
    device: str = DEVICE,
) -> float:
    """
    Trains the one-step predictor model on Henon map dataset.

    Args:
        mode (ObservabilityMode): insufficient or sufficient observability
        n_traj (int): number of trajectories in a dataset
        len_traj (int): length of each trajectory
        burn (int): number of points discarded from the beggining
            of each trajectory. See generate_dataset for more info

        hidden_size (int): number of neurons in a hidden layer
        n_epochs (int): number of epochs in training
        learning_rate (float): model's speed of learning
        seed (int): seed for weight initialization
        device (str): device used for training, e.g. "cpu" or "mps"

    Returns:
        float: best validation loss, the saved model is the one that reached it
    """

    torch.manual_seed(seed)
    rng = np.random.default_rng(TRAIN_DATA_SEED)
    X, y = generate_dataset(n_traj, len_traj, burn, mode, rng)
    val_rng = np.random.default_rng(VAL_DATA_SEED)
    X_val, y_val = generate_dataset(n_traj // 5, len_traj, burn, mode, val_rng)
    point_dim = observation_dim(mode)
    X, y = X.to(device), y.to(device)
    X_val, y_val = X_val.to(device), y_val.to(device)

    model = Predictor(point_dim=point_dim, hidden_size=hidden_size)
    model.to(device)
    loss_fn = nn.MSELoss()
    optimizer = Adam(params=model.parameters(), lr=learning_rate)

    best_val_loss = float("inf")
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    for epoch in range(n_epochs):
        model.train()
        prediction = model(X)
        loss = loss_fn(prediction, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 500 == 0:
            model.eval()
            with torch.no_grad():
                val_loss = loss_fn(model(X_val), y_val).item()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
            print(
                f"Epoch: {epoch + 1:04d} | Loss: {loss.item():.4g} | "
                f"Val loss: {val_loss:.4g}"
            )

    torch.save(best_state, model_path(mode, seed))

    return best_val_loss


if __name__ == "__main__":
    loss = []
    for mode in ("insufficient", "sufficient"):
        for seed in TRAINING_SEEDS:
            mode_loss = training(
                mode=mode,
                n_traj=1000,
                len_traj=36,
                burn=BURN_IN,
                hidden_size=HIDDEN_SIZE,
                n_epochs=10000,
                learning_rate=1e-3,
                seed=seed,
            )
            loss.append(f"{mode} seed {seed}: {mode_loss}")

    print(loss)

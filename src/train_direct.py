import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from config import (
    BURN_IN,
    DEVICE,
    DP_HIDDEN_SIZE,
    K_MAX,
    SEED,
    TRAIN_DATA_SEED,
    TRAINING_SEEDS,
    VAL_DATA_SEED,
    ObservabilityMode,
    direct_model_path,
)
from direct_predictor import DirectPredictor
from toystateworld import generate_horizon_dataset, observation_dim


def training_direct(
    mode: ObservabilityMode,
    n_traj: int = 1000,
    len_traj: int = 36,
    burn: int = BURN_IN,
    k_max: int = K_MAX,
    hidden_size: int = DP_HIDDEN_SIZE,
    n_epochs: int = 10000,
    learning_rate: float = 1e-3,
    seed: int = SEED,
    device: str = DEVICE,
) -> float:
    """
    Trains the direct predictor model on Henon map dataset.

    Args:
        mode (ObservabilityMode): insufficient or sufficient observation projection
        n_traj (int): number of trajectories in a dataset
        len_traj (int): length of each trajectory
        burn (int): number of points discarded from the beggining
            of each trajectory. See generate_dataset for more info
        k_max (int): maximum number of steps in prediction

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
    X, y = generate_horizon_dataset(n_traj, len_traj, burn, k_max, mode, rng)
    val_rng = np.random.default_rng(VAL_DATA_SEED)
    X_val, y_val = generate_horizon_dataset(
        n_traj // 5, len_traj, burn, k_max, mode, val_rng
    )
    point_dim = observation_dim(mode)
    X, y = X.to(device), y.to(device)
    X_val, y_val = X_val.to(device), y_val.to(device)

    model = DirectPredictor(k_max=k_max, point_dim=point_dim, hidden_size=hidden_size)
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

    torch.save(best_state, direct_model_path(mode, seed))

    return best_val_loss


if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    loss = []
    for mode in ("insufficient", "sufficient"):
        for seed in TRAINING_SEEDS:
            mode_loss = training_direct(
                mode=mode,
                n_traj=1000,
                len_traj=36,
                burn=BURN_IN,
                k_max=K_MAX,
                hidden_size=DP_HIDDEN_SIZE,
                n_epochs=20000,
                learning_rate=1e-3,
                seed=seed,
                device=device,
            )
            loss.append(f"{mode} seed {seed}: {mode_loss}")

    print(loss)

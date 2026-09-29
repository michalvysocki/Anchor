import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from config import HIDDEN_SIZE, SEED, ObservabilityMode, model_path
from model import Predictor
from toystateworld import generate_dataset, observation_dim

np.random.seed(seed=SEED)
torch.manual_seed(seed=SEED)


def training(
    mode: ObservabilityMode,
    n_traj: int = 1000,
    len_traj: int = 20,
    burn: int = 5,
    hidden_size: int = HIDDEN_SIZE,
    n_epochs: int = 10000,
    learning_rate: float = 1e-3,
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

    Returns:
        float: predictor's final loss
    """

    X, y = generate_dataset(n_traj, len_traj, burn, mode)
    point_dim = observation_dim(mode)

    model = Predictor(point_dim=point_dim, hidden_size=hidden_size)
    loss_fn = nn.MSELoss()
    optimizer = Adam(params=model.parameters(), lr=learning_rate)

    for epoch in range(n_epochs):
        model.train()
        prediction = model(X)
        loss = loss_fn(prediction, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 500 == 0:
            print(f"Epoch: {epoch + 1:04d} | Loss: {loss.item():.4g}")

    torch.save(model.state_dict(), model_path(mode))

    return loss.item()


if __name__ == "__main__":
    loss = []
    for mode in ("insufficient", "sufficient"):
        mode_loss = training(
            mode=mode,
            n_traj=1000,
            len_traj=20,
            burn=5,
            hidden_size=HIDDEN_SIZE,
            n_epochs=10000,
            learning_rate=1e-3,
        )
        loss.append(f"{mode}: {mode_loss}")

    print(loss)

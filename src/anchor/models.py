import torch
from torch import nn


class HorizonMLP(nn.Module):
    """
    Horizon-conditioned predictor F(z, k): an MLP taking a point and a
    one-hot horizon. Used both for direct prediction F(z, k) and, through
    UnitStep, for rollouts F(., 1)^k.

    Attributes:
        hidden_layer1 (torch.nn.Linear): Linear hidden layer
        activation1 (torch.nn.Tanh): Tanh activation layer
        hidden_layer2 (torch.nn.Linear): Linear hidden layer
        activation2 (torch.nn.Tanh): Tanh activation layer
        output_layer (torch.nn.Linear): Linear output layer
    """

    def __init__(self, k_max: int, point_dim: int, hidden_size: int):
        """
        Initialize HorizonMLP object.

        Args:
            k_max (int): maximum length of a trajectory
            point_dim (int): trajectory's point dimensions
            hidden_size (int): size of the hidden layers
        """

        super().__init__()

        self.hidden_layer1 = nn.Linear(k_max + point_dim, hidden_size)
        self.activation1 = nn.Tanh()

        self.hidden_layer2 = nn.Linear(hidden_size, hidden_size)
        self.activation2 = nn.Tanh()
        self.output_layer = nn.Linear(hidden_size, point_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Executes a forward pass through a neural network.

        Args:
            x (torch.Tensor): input tensor with data package

        Returns:
            torch.Tensor: output tensor with model predictions
        """

        x = self.hidden_layer1(x)
        x = self.activation1(x)

        x = self.hidden_layer2(x)
        x = self.activation2(x)

        x = self.output_layer(x)

        return x


def horizon_input(z: torch.Tensor, k: torch.Tensor, k_max: int) -> torch.Tensor:
    """
    Concatenates points with one-hot encoded horizons.

    Args:
        z (torch.Tensor): points of size (batch, point_dim)
        k (torch.Tensor): horizons from 1 to k_max of size (batch,)
        k_max (int): maximum step

    Returns:
        torch.Tensor: model input of size (batch, point_dim + k_max)
    """

    one_hot = nn.functional.one_hot(k - 1, k_max).to(z.dtype)

    return torch.cat([z, one_hot], dim=1)


def predict(
    model: nn.Module, z: torch.Tensor, k: torch.Tensor, k_max: int
) -> torch.Tensor:
    """
    Predicts the state k steps ahead with a horizon-conditioned model.

    Args:
        model (nn.Module): model taking points with one-hot horizons
        z (torch.Tensor): points of size (batch, point_dim)
        k (torch.Tensor): horizons from 1 to k_max of size (batch,)
        k_max (int): maximum step

    Returns:
        torch.Tensor: predicted points of size (batch, point_dim)
    """

    return model(horizon_input(z, k, k_max))


class UnitStep(nn.Module):
    """
    Uses a horizon-conditioned model F(z, k) as a one-step model F(z, 1),
    so every variant trains and rolls out the same network.
    """

    def __init__(self, model: nn.Module, k_max: int):
        super().__init__()
        self.model = model
        self.k_max = k_max

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        k = torch.ones(len(z), dtype=torch.int64, device=z.device)
        return predict(self.model, z, k, self.k_max)

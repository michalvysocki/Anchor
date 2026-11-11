import torch
from torch import nn


class Predictor(nn.Module):
    """
    Class representing a one-step predictor model.

    Attributes:
        hidden_layer (torch.nn.Linear): Linear hidden layer
        activation (torch.nn.Tanh): Tanh activation layer
        output_layer (torch.nn.Linear): Linear output layer
    """

    def __init__(self, point_dim: int, hidden_size: int = 64):
        """
        Initialize Predictor object.

        Args:
            point_dim (int): trajectory's point dimensions
            hidden_size (int): size of the hidden layers, default
            value is 64 because of experiments
        """

        super().__init__()

        self.hidden_layer = nn.Linear(point_dim, hidden_size)
        self.activation = nn.Tanh()
        self.output_layer = nn.Linear(hidden_size, point_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Executes a forward pass through a neural network.

        Args:
            x (torch.Tensor): input tensor with data package

        Returns:
            torch.Tensor: output tensor with model predictions
        """

        x = self.hidden_layer(x)
        x = self.activation(x)
        x = self.output_layer(x)

        return x

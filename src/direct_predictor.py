import torch
from torch import nn


class DirectPredictor(nn.Module):
    """
    Class representing a direct head predictor model.

    Attributes:
        hidden_layer1 (torch.nn.Linear): Linear hidden layer
        activation1 (torch.nn.Tanh): Tanh activation layer
        hidden_layer2 (torch.nn.Linear): Linear hidden layer
        activation2 (torch.nn.Tanh): Tanh activation layer
        output_layer (torch.nn.Linear): Linear output layer
    """

    def __init__(self, k_max: int, point_dim: int, hidden_size: int):
        """
        Initialize DirectPredictor object.

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

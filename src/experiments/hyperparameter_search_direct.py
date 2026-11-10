import numpy as np

from train_direct import training_direct

if __name__ == "__main__":
    loss_history = []
    modes = ("insufficient", "sufficient")
    configs = [
        (1000, 128, 20000),
        (1000, 256, 20000),
        #(1000, 64, 20000),
        #(1000, 64, 30000),
    ]
    n_repeats = 10
    learning_rate = 1e-3

    runs = [(mode, config) for mode in modes for config in configs]

    for mode, config in runs:
        loss_pair = []
        n_traj, hidden_size, n_epochs = config

        for rep in range(n_repeats):
            loss = training_direct(
                mode=mode,
                n_traj=n_traj,
                hidden_size=hidden_size,
                n_epochs=n_epochs,
                learning_rate=learning_rate,
                seed=rep,
            )
            loss_pair.append(loss)

        loss_history.append(loss_pair)

    for (mode, config), losses in zip(runs, loss_history):
        mean = np.mean(losses)
        std = np.std(losses)

        print(
            f"Mode: {mode} | "
            f"Number of trajectories: {config[0]} | "
            f"Hidden layer size: {config[1]} | "
            f"Number of repeats: {n_repeats}| "
            f"Number of epochs: {config[2]} | "
            f"Mean val loss: {mean} | "
            f"Val loss std: {std} | "
        )

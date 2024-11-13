import os

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau

from datetime import datetime
from typing import Optional, Tuple

from src.data_loader import get_data_loader
from src.logging_config import setup_logger


logger = setup_logger(__name__)


def train_model(
    model_type: nn.Module,
    lr: float = 0.001,
    batch_size: int = 32,
    epochs: int = 10,
    model_path: str = "logs/model",
    data_dir: str = "data/train",
    device: Optional[str] = "cuda",
) -> str:
    logger.info("Training the model")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_dir = os.path.join(model_path, timestamp)
    os.makedirs(model_dir, exist_ok=True)

    train_loader = get_data_loader(data_dir, batch_size)
    model = model_type().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, "min", patience=3, verbose=True)

    loss_values = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device).float()
            optimizer.zero_grad()

            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        loss_values.append(avg_loss)
        scheduler.step(avg_loss)

        logger.info(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

    model_file_path = os.path.join(model_dir, f"model.pth")
    torch.save(model.state_dict(), model_file_path)
    logger.info(f"Model saved to {model_file_path}")

    plt.figure()
    plt.plot(
        range(1, epochs + 1), loss_values, marker="o", color="b", label="Training Loss"
    )
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.legend()

    loss_plot_path = os.path.join(model_dir, f"loss_plot.png")
    plt.savefig(loss_plot_path)
    logger.info(f"Training loss plot saved to {loss_plot_path}")
    plt.close()

    return model_dir


if __name__ == "__main__":
    pass

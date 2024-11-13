import os
import json
import torch
import numpy as np

import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt

from typing import Tuple
from sklearn.metrics import accuracy_score, confusion_matrix

from src.data_loader import get_data_loader
from src.logging_config import setup_logger

logger = setup_logger(__name__)


def eval_model(
    model_type: nn.Module, model_dir: str, data_dir: str = "data/test"
) -> Tuple[dict, str]:
    logger.info("Initializing model.")
    model = model_type().cuda()

    model_files = [file for file in os.listdir(model_dir) if file.endswith(".pth")]
    if not model_files:
        logger.error("No model file found in the specified directory.")
        raise FileNotFoundError("No model file found in the specified directory.")
    model_path = os.path.join(model_dir, model_files[0])

    logger.info(f"Loading model from {model_path}.")
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    logger.info("Loading test data.")
    test_loader = get_data_loader(data_dir, batch_size=32)

    all_preds = []
    all_labels = []

    logger.info("Starting model evaluation.")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.cuda(), labels.cuda()

            outputs = model(inputs)
            preds = (outputs.squeeze() > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Confusion Matrix: \n{cm}")

    results = {
        "accuracy": accuracy,
        "confusion_matrix": cm.tolist(),
    }
    results_path = os.path.join(model_dir, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    logger.info(f"Results saved to {results_path}.")

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Benign", "Malignant"],
        yticklabels=["Benign", "Malignant"],
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix\nAccuracy: {accuracy:.4f}")

    confusion_matrix_path = os.path.join(model_dir, "confusion_matrix.png")
    plt.savefig(confusion_matrix_path)
    plt.close()

    logger.info(f"Confusion matrix saved to {confusion_matrix_path}.")
    logger.info(f"Evaluation results saved in {model_dir}")

    return results, confusion_matrix_path


def display_model_results(model_dir: str):
    logger.info(f"Displaying results from {model_dir}")

    json_file = os.path.join(model_dir, "evaluation_results.json")
    if not os.path.exists(json_file):
        logger.error("Results JSON file not found.")
        return

    with open(json_file, "r") as f:
        results = json.load(f)

    accuracy = results.get("accuracy", "N/A")
    confusion_matrix = np.array(results.get("confusion_matrix", []))

    if confusion_matrix.size == 0:
        logger.error("Confusion matrix data not found.")
        return

    print(f"\033[92mAccuracy: {accuracy * 100:.2f}%\033[0m")

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Benign", "Malignant"],
        yticklabels=["Benign", "Malignant"],
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    loss_plot_path = os.path.join(model_dir, "loss_plot.png")
    if os.path.exists(loss_plot_path):
        loss_plot_img = plt.imread(loss_plot_path)
        plt.figure(figsize=(6, 5))
        plt.imshow(loss_plot_img)
        plt.title("Training Loss Over Epochs")
        plt.axis("off")  # Turn off axes
        plt.show()
    else:
        logger.error("Loss plot not found.")

    logger.info("Model results displayed successfully.")


if __name__ == "__main__":
    pass

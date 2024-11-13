import os

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from src.logging_config import setup_logger

logger = setup_logger(__name__)


class BreastCancerDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        logger.debug("Initializing BreastCancerDataset")

        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

        logger.info("BreastCancerDataset initialized")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]

        image = Image.open(img_path).convert("L")
        label = self.labels[index]

        if self.transform:
            image = self.transform(image)

        return image, label


def get_data_loader(data_dir: str, batch_size: int) -> DataLoader:
    logger.debug(f"Getting data loader for {data_dir} with batch size {batch_size}")
    image_paths, labels = [], []

    for file in os.listdir(f"{data_dir}/Benign"):
        image_paths.append(f"{data_dir}/Benign/{file}")
        labels.append(1)

    for file in os.listdir(f"{data_dir}/Malignant"):
        image_paths.append(f"{data_dir}/Malignant/{file}")
        labels.append(0)

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
        ]
    )

    dataset = BreastCancerDataset(image_paths, labels, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    logger.info(f"Data loader created for {data_dir} with batch size {batch_size}")
    return data_loader


if __name__ == "__main__":
    pass

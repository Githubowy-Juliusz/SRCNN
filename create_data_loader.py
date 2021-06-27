import numpy as np
from torch.utils.data import DataLoader
from dataset import ImageDataset


def create_data_loader(images: np.ndarray, validation: bool, batch_size=900) -> DataLoader:
	dataset = ImageDataset(images, validation)
	loader = DataLoader(dataset, batch_size=batch_size)
	return loader
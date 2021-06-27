import random
import torch
from torch.utils.data import Dataset
import transposers
from images_utils import create_low_res_image


class ImageDataset(Dataset):
	def __init__(self, image_data, validation: bool):
		self.image_data = image_data
		self.validation = validation

	def __len__(self):
		return (len(self.image_data))

	def __getitem__(self, index):
		image = self.image_data[index]

		image_low_res = transposers.to_torch(create_low_res_image(image))
		images_high_res = transposers.to_torch(image)
		return (
				torch.tensor(image_low_res / 255.0, dtype=torch.float),
				torch.tensor(images_high_res / 255.0, dtype=torch.float)
		)
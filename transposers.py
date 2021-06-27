import numpy as np
#normal image: height, width, channel
#torch image: channel, height, width

def to_torch(array: np.ndarray) -> np.ndarray:
	if len(array.shape) == 4: #array of images
		return array.transpose(0, 3, 1, 2)
	return array.transpose(2, 0, 1)

def to_image(array: np.ndarray) -> np.ndarray:
	if len(array.shape) == 4:
		return array.transpose(0, 2, 3, 1)
	return array.transpose(1, 2, 0)
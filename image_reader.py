import os
import numpy as np
import cv2 as cv

class ImageReader:
	def __init__(self, directory="./DIV2K_train_HR", square_length=64):
		self.directory = directory
		self.square_length = square_length
		self._index = 0
		self._val_index_begin = 725
		self._test_index_begin = 750
		splits = 5
		indexes = np.linspace(0, self._val_index_begin, splits).astype(int)
		self._begins = indexes[:-1]
		self._ends = indexes[1:]

	def __len__(self):
		return (len(self._begins))
	
	def read_train(self) -> np.ndarray:
		idx_from = self._begins[self._index]
		idx_to = self._ends[self._index]

		self._index += 1
		if self._index >= len(self._begins):
			self._index = 0
		
		return self._read_images(idx_from, idx_to)

	def read_val(self) -> np.ndarray:
		return self._read_images(self._val_index_begin, self._test_index_begin)

	def read_test(self) -> np.ndarray:
		return self._read_images(self._test_index_begin, 800)

	def read_sample_image(self, index=797, directory="") -> np.ndarray:
		if directory == "":
			directory = self.directory
		image_name = sorted(os.listdir(directory))[index]
		image = cv.imread(f"{directory}/{image_name}")
		image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
		return image

	def _read_images(self, idx_from, idx_to) -> np.ndarray:
		imgs = []
		for path in sorted(os.listdir(self.directory))[idx_from:idx_to]:
			path = f"{self.directory}/{path}"
			if path.endswith(".jpg") or path.endswith(".png"):
				imgs.extend(self._read_and_preprocess(path))

		return np.array(imgs)

	def _read_and_preprocess(self, path) -> np.ndarray:
		img = cv.imread(path)
		img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
		imgs = self._cut_images(img)
		return imgs

	def _cut_images(self, img) -> np.ndarray:
		square_length = self.square_length
		height = img.shape[0]
		width = img.shape[1]
		imgs = []
		for x in range(0, width, square_length):
			for y in range(0, height, square_length):
				cut_img = img[y:y + square_length, x:x + square_length, :]
				if(cut_img.shape == (square_length, square_length, 3)):
					imgs.append(cut_img)

		return np.array(imgs)
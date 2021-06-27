import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from images_utils import add_noise, add_salt_and_pepper


def plot_noise(image: np.ndarray):
	fig = plt.figure(figsize=(20, 40))
	fig.add_subplot(3, 1, 1)
	plt.imshow(image)
	fig.add_subplot(3, 1, 2)
	noise_img = add_noise(image)
	plt.imshow(noise_img)
	fig.add_subplot(3, 1, 3)
	salt_pepper_img = add_salt_and_pepper(image)
	plt.imshow(salt_pepper_img)
	plt.show()
	cv.imwrite("./output_images/noise.png", cv.cvtColor(noise_img, cv.COLOR_RGB2BGR))
	cv.imwrite("./output_images/salt.png", cv.cvtColor(salt_pepper_img, cv.COLOR_RGB2BGR))
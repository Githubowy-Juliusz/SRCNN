import numpy as np
import cv2 as cv
from psnr import psnr_numpy


def create_low_res_images(images: np.ndarray) -> np.ndarray:
	images_low_res = tuple(create_low_res_image(image) for image in images)
	return np.array(images_low_res)

def create_low_res_image(image: np.ndarray) -> np.ndarray:
	return resize_up(resize_down(image))

def calculate_psnr_with_bicubic(images: np.ndarray) -> float:
	low_res_images = create_low_res_images(images) / 255.0
	images = images / 255.0
	return psnr_numpy(low_res_images, images)

def add_salt_and_pepper(image: np.ndarray) -> np.ndarray:
	pepper = np.random.random(image.shape[:2]) > 0.01
	salt = (np.random.random(image.shape[:2]) > 0.99) * 254 + 1
	pepper = np.stack((pepper, pepper, pepper), axis=2)
	salt = np.stack((salt, salt, salt), axis=2)
	img = image * pepper
	img = img * salt
	img[img > 255] = 255
	return img.astype(np.uint8)

def add_noise(image: np.ndarray) -> np.ndarray:
	noise = np.random.random(image.shape) * 50 - 25
	img = image + noise
	img[img > 255] = 255
	img[img < 0] = 0
	return img.astype(np.uint8)

def resize_down(image) -> np.ndarray:
	return cv.resize(image, (image.shape[1] // 2, image.shape[0] // 2), interpolation=cv.INTER_CUBIC)

def resize_up(image) -> np.ndarray:
	return cv.resize(image, (image.shape[1] * 2, image.shape[0] * 2), interpolation=cv.INTER_CUBIC)
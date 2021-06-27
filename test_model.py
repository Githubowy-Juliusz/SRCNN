import os
import datetime as dt
import numpy as np
import cv2 as cv
import torch
import matplotlib.pyplot as plt
import transposers
from psnr import psnr_numpy
from images_utils import create_low_res_image, calculate_psnr_with_bicubic, add_salt_and_pepper, add_noise
from images_utils import resize_down, resize_up


def calculate_test_psnr(model, images, device):
	psnr_list = []
	for img in images:
		low_res_img = create_low_res_image(img)
		predicted_img = predict(model, low_res_img, device)

		psnr_ = psnr_numpy(predicted_img / 255.0, img / 255.0)
		psnr_list.append(psnr_)
	return f"model psnr:{np.array(psnr_list).mean()} bicubic psnr:{calculate_psnr_with_bicubic(images)}"

def calculate_test_psnr_with_salt_and_pepper(model, images, device):
	psnr_list = []
	psnr_bicubic_list = []
	for img in images:
		low_res_img = resize_up(add_salt_and_pepper(resize_down(img)))
		predicted_img = predict(model, low_res_img, device)

		psnr_ = psnr_numpy(predicted_img / 255.0, img / 255.0)
		psnr_bicubic = psnr_numpy(low_res_img / 255.0, img / 255.0)
		psnr_list.append(psnr_)
		psnr_bicubic_list.append(psnr_bicubic)
	return f"model psnr:{np.array(psnr_list).mean()} bicubic psnr:{np.array(psnr_bicubic_list).mean()}"

def calculate_test_psnr_with_noise(model, images, device):
	psnr_list = []
	psnr_bicubic_list = []
	for img in images:
		low_res_img = resize_up(add_noise(resize_down(img)))
		predicted_img = predict(model, low_res_img, device)

		psnr_ = psnr_numpy(predicted_img / 255.0, img / 255.0)
		psnr_bicubic = psnr_numpy(low_res_img / 255.0, img / 255.0)
		psnr_list.append(psnr_)
		psnr_bicubic_list.append(psnr_bicubic)
	return f"model psnr:{np.array(psnr_list).mean()} bicubic psnr:{np.array(psnr_bicubic_list).mean()}"

def test_model(model: torch.nn.Module, image: np.ndarray, device: str):
	low_res_image = create_low_res_image(image)
	predicted_image = predict(model, low_res_image, device)

	joined_image = _create_joined_image(low_res_image, image, predicted_image)
	detail_image = _create_detail_image(low_res_image, image, predicted_image)

	directory = "./output_images"
	if not os.path.exists(directory):
		os.mkdir(directory)

	now = str(dt.datetime.now())
	cv.imwrite(f"{directory}/detail_{now}.png", cv.cvtColor(detail_image, cv.COLOR_RGB2BGR))
	cv.imwrite(f"{directory}/joined_{now}.png", cv.cvtColor(joined_image, cv.COLOR_RGB2BGR))

	plt.figure(figsize=(40, 40))
	plt.imshow(detail_image)
	plt.show()

	plt.figure(figsize=(40, 40))
	plt.imshow(joined_image)
	plt.show()

def predict(model: torch.nn.Module, image: np.ndarray, device: str) -> np.ndarray:
	image = transposers.to_torch(image) / 255.0
	model.eval()
	with torch.no_grad():
		input_image = torch.tensor(image, dtype=torch.float).to(device)
		input_image = input_image.unsqueeze(0)
		predicted_image = model(input_image)
	
	predicted_image = predicted_image.cpu().detach().numpy()
	predicted_image = transposers.to_image(predicted_image[0])
	predicted_image *= 255
	predicted_image[predicted_image > 255] = 255
	predicted_image[predicted_image < 0 ] = 0
	return predicted_image.astype(np.uint8)

def _create_joined_image(low_res_image, high_res_image, predicted_image) -> np.ndarray:
	divider = high_res_image.shape[1] // 3

	final_image = np.zeros(shape=high_res_image.shape, dtype=np.uint8)
	final_image[:, 0:divider, :] = low_res_image[:, 0:divider, :]
	final_image[:, divider:divider * 2, :] = predicted_image[:, divider:divider * 2, :]
	final_image[:, divider * 2:, :] = high_res_image[:, divider * 2:, :]
	final_image[:, divider, :] = 0
	final_image[:, divider * 2, :] = 0

	return final_image

def _create_detail_image(low_res_image, high_res_image, predicted_image) -> np.ndarray:
	indexer = (slice(500, 800), slice(600, 1100), slice(0, 4))
	return np.vstack((low_res_image[indexer], predicted_image[indexer], high_res_image[indexer]))
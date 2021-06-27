import torch
import numpy as np

def _psnr_numpy(prediction: np.ndarray, target: np.ndarray, max_value_squared: float = 1.0) -> float:
	difference = prediction - target
	mse = ((difference) ** 2).mean()
	if mse == 0.0:
		return 100
	return 10 * np.log10(max_value_squared / mse)

def psnr_numpy(prediction: np.ndarray, target: np.ndarray, max_value_squared: float = 1.0) -> float:
	if len(prediction.shape) == 3:
		return _psnr_numpy(prediction, target, max_value_squared)

	psnr_list = []
	for _prediction, _target in zip(prediction, target):
		psnr_list.append(_psnr_numpy(_prediction, _target, max_value_squared))
	return np.array(psnr_list).mean()

def _psnr_torch(prediction: torch.Tensor, target: torch.Tensor, max_value_squared: float = 1.0) -> float:
	prediction = prediction.detach().cpu().numpy()
	target = target.detach().cpu().numpy()
	return psnr_numpy(prediction, target, max_value_squared)

def psnr(prediction: torch.Tensor, target: torch.Tensor, max_value_squared: float = 1.0) -> float:
	if len(prediction.shape) == 3:
		return _psnr_torch(prediction, target, max_value_squared)

	psnr_list = []
	for _prediction, _target in zip(prediction, target):
		psnr_list.append(_psnr_torch(_prediction, _target, max_value_squared))
	return np.array(psnr_list).mean()

def psnr_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
	difference = prediction - target
	mse = ((difference) ** 2).mean()
	return -10 * torch.log10(1.0 / mse)
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from psnr import psnr
from pretty_elapsed_time import pretty_elapsed_time


class Trainer:
	def __init__(self, model: nn.Module, train_loader: DataLoader,
		val_loader: DataLoader, device: str, model_saver):
		self.model = model
		self.train_loader = train_loader
		self.val_loader = val_loader
		self.device = device
		self.loss_function = nn.MSELoss()
		self.model_saver = model_saver

	def train(self, epochs: int, lr=0.001):
		start_time = time.time()

		train_loss, val_loss = np.empty(epochs), np.empty(epochs)
		val_psnr = np.empty(epochs)
		
		optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.1)

		for epoch in range(epochs):
			print(f"Dataset epoch {epoch + 1}/{epochs}")

			train_epoch_loss = self._train(optimizer)
			val_epoch_loss, val_epoch_psnr = self._validate()

			train_loss[epoch] = train_epoch_loss
			val_loss[epoch] = val_epoch_loss
			val_psnr[epoch] = val_epoch_psnr

		time_elapsed = time.time() - start_time
		print(f"Finished dataset in: {pretty_elapsed_time(time_elapsed)}")
		return train_loss, val_loss, val_psnr

	def _train(self, optimizer):
		self.model.train()
		loss_list = []
		for data in self.train_loader:
			image_data = data[0].to(self.device)
			target = data[1].to(self.device)
			
			prediction = self.model(image_data)
			loss = self.loss_function(prediction, target)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			loss_list.append(loss.item())

		mean_loss = np.array(loss_list).mean()
		return mean_loss

	def _validate(self):
		self.model.eval()
		loss_list = []
		psnr_list = []
		with torch.no_grad():
			for data in self.val_loader:
				image_data = data[0].to(self.device)
				target = data[1].to(self.device)
				
				prediction = self.model(image_data)
				loss = self.loss_function(prediction, target)

				loss_list.append(loss.item())
				psnr_list.append(psnr(prediction, target))

		mean_loss = np.array(loss_list).mean()
		mean_psnr = np.array(psnr_list).mean()
		self.model_saver.conditional_save(self.model, mean_psnr)
		return mean_loss, mean_psnr

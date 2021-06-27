import os
from datetime import datetime
import torch

class ModelSaver:
	def __init__(self, save_directory: str, last_val_psnr: float, current_epoch: float):
		self.save_directory = save_directory
		if not os.path.isdir(self.save_directory):
			os.mkdir(self.save_directory)
		self.last_val_psnr = last_val_psnr
		self.current_epoch = current_epoch
	
	def save(self, model, psnr: float):
		now = datetime.now()
		model_name = f"srcnn_{now}_epoch_{self.current_epoch}_psnr_{psnr:.3f}"
		model_path = f"{self.save_directory}/{model_name}"
		try:
			torch.save(model.state_dict(), model_path)
		except Exception as e:
			print(f"Exception thrown when saving model: {e}")

	def conditional_save(self, model, psnr: float):
		if psnr >= self.last_val_psnr:
			self.last_val_psnr = psnr
			self.save(model, psnr)
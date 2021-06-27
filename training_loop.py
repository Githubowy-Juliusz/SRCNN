import time
import numpy as np
from create_data_loader import create_data_loader
from image_reader import ImageReader
from model_saver import ModelSaver
from trainer import Trainer
from pretty_elapsed_time import pretty_elapsed_time

def training_loop(trainer: Trainer, image_reader: ImageReader,
		model_saver: ModelSaver, epochs:int, 
		already_trained_epochs: int, epochs_per_dataset=5, lr=0.001):
	start_time = time.time()
	
	array_size = epochs * len(image_reader) * epochs_per_dataset
	train_losses = np.empty(array_size)
	val_losses = np.empty(array_size)
	val_psnrs = np.empty(array_size)

	for actual_epoch in range(epochs):
		print(f"Actual epoch: {actual_epoch + 1}/{epochs}")
		for dataset_number in range(0, len(image_reader)):
			print(f"Dataset {dataset_number + 1}/{len(image_reader)}")
			images = image_reader.read_train()
			trainer.train_loader = create_data_loader(images, validation=False)

			model_saver.current_epoch = (actual_epoch) * epochs_per_dataset\
				+ already_trained_epochs + (dataset_number + 1) / len(image_reader)\
				* epochs_per_dataset
			train_loss, val_loss, val_psnr = trainer.train(epochs_per_dataset, lr=lr)

			current_index = (actual_epoch * len(image_reader) * epochs_per_dataset
				+ dataset_number * epochs_per_dataset)
			to_index = current_index + epochs_per_dataset

			train_losses[current_index:to_index] = train_loss
			val_losses[current_index:to_index] = val_loss
			val_psnrs[current_index:to_index] = val_psnr

		model_saver.current_epoch = (actual_epoch + 1) * epochs_per_dataset\
			+ already_trained_epochs
		model_saver.save(trainer.model, val_psnrs[-1])

	time_elapsed = time.time() - start_time
	time_elapsed = pretty_elapsed_time(time_elapsed)
	print(f"Finished in: {time_elapsed}")
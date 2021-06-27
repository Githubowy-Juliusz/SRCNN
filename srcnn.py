import torch.nn as nn
import torch.nn.functional as F


class SRCNN(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4, padding_mode="replicate")
		self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0, padding_mode="replicate")
		self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2, padding_mode="replicate")
		#self.conv4 = nn.Conv2d(16, 3, kernel_size=3, padding=1, padding_mode="replicate")

	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = F.relu(self.conv3(x))
		#x = F.relu(self.conv4(x))
		return x
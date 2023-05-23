# pip install vit-pytorch

import os

import torch
from torch.optim.lr_scheduler import StepLR
from torchvision.datasets import FakeData
from torchvision.transforms import CenterCrop
from vit_pytorch.cct import CCT
from vit_pytorch import SimpleViT
from text_overlay_dataset import TextOverlayDataset

# logging
from verta import Client
client = Client(email=os.environ["VERTA_EMAIL"], dev_key=os.environ["VERTA_DEV_KEY"])
proj = client.set_project("Optical Character Recognition")
expt = client.set_experiment("Vision Transformer + LSTM")
run = client.set_experiment_run("Random Noise + Random Text")
config = dict(
	run_name="Random Noise + Random Text",
	input_image_width = 512,
	patch_size = 16,
	transformer_latent_dim = 512,
	transformer_output_dim = 1024,
	vit_output_dim = 2048,
	lstm_latent_dim = 2048,
)
#run.log_hyperparameter("transformer_latent_dim", 512)

# Our text dataset is randomly generated.
def generate_quadgrams():
	import gzip
	import csv
	from collections import deque
	from unidecode import unidecode
	csv.field_size_limit(65535*8)
	gin = gzip.open("/mnt/synology/wikipedia_utf8_filtered_20pageviews.csv.gz", 'rt')
	cin = csv.reader(gin)
	quadgrams = dict()
	for index, line in cin:
	    buffer = deque(maxlen=4)
	    for c in line:
	        gram = "".join(buffer)
	        if gram not in quadgrams:
	            quadgrams[gram] = dict()  # next letter to count
	        if c not in quadgrams[gram]:
	            quadgrams[gram][c] = 0
	        quadgrams[gram][c] += 1
	        buffer.append(c)
	    if len(quadgrams) > 10:
		        break
	return quadgrams


class OCRModel(torch.nn.Module):
	def __init__(self, 
			input_image_width: int = 512, 
			patch_size: int = 16, 
			transformer_latent_dim: int = 512, 
			transformer_output_dim: int = 1024, 
			vit_output_dim: int = 2048,
			lstm_latent_dim: int = 2048,
	):
		self.vit = SimpleViT(
			image_size = image_size,
			patch_size = patch_size,
			heads = 16,
			depth = 6,
			mlp_dim = transformer_latent_dim,
			dim = transformer_output_dim,
			num_classes = vit_output_dim,
		)
		self.rnn = torch.nn.LSTM(vit_output_dim, lstm_datent_dim, batch_first=True)
		self.output = torch.nn.Linear(vit_output_dim, 255)

	def forward(self, image_in, return_logits: bool = False, cap_length: int = -1):
		embedding = self.vit(image_In)
		


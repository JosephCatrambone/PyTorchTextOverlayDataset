# HACK for running without having text_overlay_dataset installed via pip.
import sys
sys.path.append("/home/joseph/PythonSource/PyTorchTextOverlayDataset/src/")

import math
import random
import os
from glob import glob
from typing import List, Optional

import torch
import torchvision
from PIL import Image
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.fakedata import FakeData
from torchvision.models import resnet50
from torchvision.transforms import RandomResizedCrop, ToPILImage, PILToTensor
from torchvision.transforms.functional import pil_to_tensor
from text_overlay_dataset import TextOverlayDataset
from tqdm import tqdm


# logging
config = dict(
	run_name="Random Noise + Random Text",
	vision_output_dim = 1024,
	model_output_characters = 5,  # How many letters do we want in our range?
	model_output_dim = 255,
	batch_size = 16,
	num_epochs = 1000,
	learning_rate=1e-2,
)
try:
	from torch.utils.tensorboard import SummaryWriter
	writer = SummaryWriter("runs/train_resnet_trigrams")
except ModuleNotFoundError:
	print("tensorboard not found.  Ignoring TB writer.")
	writer = None


class RandomTextDataset(Dataset):
	def __init__(self, example_length: int, dataset_length: int, charset: Optional[str] = None):
		self.example_length = example_length
		self.charset = charset or "".join([chr(x) for x in range(ord(' '), ord('~'))])
		self.dataset_length = dataset_length
		
	def __len__(self) -> int:
		return self.dataset_length
	
	def __getitem__(self, idx):
		#return "hi!"
		return "".join(random.choice(self.charset) for _ in range(self.example_length))


class ImageDataset(Dataset):
	def __init__(self, path_glob: str):
		self.image_filenames = glob(path_glob)

	def __len__(self):
		return len(self.image_filenames)

	def __getitem__(self, idx):
		return Image.open(self.image_filenames[idx]).convert("RGB")


def strings_to_one_hot(texts: List[str], padding_char_idx: int = 0, max_length: Optional[int] = None):
	"""Given a list of strings, return a tensor of shape [len(texts), max_length, 255]."""
	if max_length is None:
		max_length = 0
		for t in texts:
			max_length = max(max_length, len(t))
	outputs = list()
	for t in texts:
		out = torch.zeros((max_length, 255))
		for letter_idx, letter in enumerate(t[:max_length]):
			if ord(letter) > 255:
				letter = chr(padding_char_idx)
			out[letter_idx,ord(letter)] = 1.0
		if len(t) < max_length-1:
			for letter_idx in range(len(t), max_length-1):
				out[letter_idx,padding_char_idx] = 1.0
		outputs.append(out)
	return torch.stack(outputs)


def logits_to_strings(tensors, padding_char_idx: int = 0, replacement_pad_character: int = 32, truncate_on_pad: bool = True) -> List[str]:
	batch_by_longs = torch.argmax(tensors, axis=-1)
	return one_hot_to_strings(batch_by_longs, padding_char_idx, replacement_pad_character, truncate_on_pad)


def one_hot_to_strings(one_hot, padding_char_idx: int = 0, replacement_pad_character: int = 32, truncate_on_pad: bool = True) -> List[str]:
	all_strings = list()
	for line_idx in range(one_hot.shape[0]):
		s = ""
		for char_idx in range(one_hot.shape[1]):
			ordinal = one_hot[line_idx, char_idx]
			if ordinal == padding_char_idx:
				if truncate_on_pad:
					break
				else:
					ordinal = replacement_pad_character
			s += chr(ordinal)
		all_strings.append(s)
	return all_strings


class OCRModel(torch.nn.Module):
	def __init__(self, 
			vision_output_dim: int = 1024,
			model_output_count: int = 5,
			model_output_dim: int = 255,
	):
		super().__init__()
		self.vision_model = resnet50(weights=None)
		vision_model_fc_feature_count = self.vision_model.fc.in_features
		self.vision_model.fc = torch.nn.Linear(in_features=vision_model_fc_feature_count, out_features=vision_output_dim)
		self.output_heads = torch.nn.ModuleList([
			torch.nn.Linear(in_features=vision_output_dim, out_features=model_output_dim) for _ in range(model_output_count)
		])
		self.vision_activation = torch.nn.SiLU()
		#self.output_activation = torch.nn.SiLU()
		#self.output_activation = torch.nn.LogSoftmax(dim=-1)  # DO NOT RUN THIS BEFORE torch.stack OR THE DIM WILL BE WRONG!  Use if we have NLLLoss.
		self.output_activation = torch.nn.Softmax(dim=-1)
		#self.output_activation = torch.nn.Identity()
		# Do we want also output_to_output?

	def forward(self, image_in, return_logits: bool = False, batch_first: bool = True):
		# If return_logits is True, the return shape is (cap_length, batch_size, model_output_dim)
		# If return_logits is False, the return shape is (cap_length, batch_size)
		# Assume image_in size is 3, 224, 224.
		assert image_in[0].shape == (3, 224, 224)
		embeddings = self.vision_activation(self.vision_model(image_in))
		logits_out = list()
		for output_head in self.output_heads:
			# output = output_head(embeddings)
			output = output_head(embeddings)
			logits_out.append(output)
		logits_out = torch.stack(logits_out)  # Shape: (cap_length, batch_size, model_output_dim)
		logits_out = self.output_activation(logits_out)

		# Maybe rearrange:
		if batch_first:
			logits_out = torch.swapaxes(logits_out, 0, 1)

		if return_logits:
			# We should do torch.nn.LogSoftmax here because with torch.nn.NLLLoss "The input given through a forward call is expected to contain log-probabilities of each class"
			return logits_out
		else:
			return torch.argmax(logits_out, dim=-1)

			
def custom_collate(list_of_tuples):
	batch_x = list()
	batch_y = list()
	for x, y, _ in list_of_tuples:
		batch_x.append(pil_to_tensor(x) / 255.0)
		batch_y.append(y)
	return torch.stack(batch_x), strings_to_one_hot(batch_y, max_length=config['model_output_characters'])


def train(device, model, dataset, batch_size: int = 10, num_epochs: int = 1, learning_rate: float = 1e-6):
	# For logging.
	# We only do this import locally so we don't get multiple instances from the different data-loader forks
	try:
		import wandb
		wandb.init(project="SimpleOCR", config=config)
	except ImportError:
		wandb = None

	#loss_fn = torch.nn.NLLLoss()
	#loss_fn = torch.nn.CrossEntropyLoss()
	loss_fn = torch.nn.BCELoss()
	optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
	scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=10)

	# If we want to use dataloader...
	batch_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate, num_workers=8, multiprocessing_context='spawn')

	lowest_epoch_loss = None
	for epoch in tqdm(range(0, num_epochs)):
		epoch_total_loss = 0.0
		local_step = 0
		
		model.train()
		for batch_x, batch_y in tqdm(batch_loader):
			batch_x = batch_x.to(device)
			batch_y = batch_y.to(device)
			
			# Inference:
			optimizer.zero_grad()
			preds = model(batch_x, return_logits=True)  # Return logits should be False for NLL and True for BCE.
			# NLLLoss expects (batch, class), but BCELoss requires (batch, seq, class).
			loss = loss_fn(preds, batch_y)
			loss.backward()
			optimizer.step()

			batch_loss = loss.item()
			epoch_total_loss += batch_loss
			if writer:
				writer.add_scalar("batch_loss", batch_loss, (epoch*len(dataset))+local_step)
				if (local_step//batch_size) % 10 == 0:
					writer.add_image("sample_image", batch_x[0], epoch*len(dataset)+local_step)
					#writer.add_text("sample_text", one_hot_to_strings(batch_y[0]), epoch*len(dataset)+local_step)
			if wandb:
				wandb.log({"batch_loss": batch_loss, "step": (epoch*len(dataset))+local_step})
			local_step += batch_size

		# Compute our performance metrics for the learning rate and determine if we're ready to downshift.
		epoch_mean_loss = epoch_total_loss / float(len(dataset))
		scheduler.step(epoch_mean_loss)

		# Maybe save the model at this point:
		if lowest_epoch_loss is None or epoch_mean_loss < lowest_epoch_loss:
			lowest_epoch_loss = epoch_mean_loss
			torch.save(model, os.path.join("checkpoints", f"ckpt_{epoch}_loss_{epoch_mean_loss}.pt"))

		# Validation:
		print(epoch_total_loss)
		model.eval()
		composite_image, text, _ = dataset[0]
		with torch.no_grad():
			preds = model(pil_to_tensor(composite_image).to(device).unsqueeze(0)/255.0, return_logits=False)
			pred_text = one_hot_to_strings(preds.to('cpu'))
			print(f"Ground truth: '{text}'")
			print(f"Model says: '{pred_text}'")


def main():
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = OCRModel(vision_output_dim=config['vision_output_dim'], model_output_count=config['model_output_characters'], model_output_dim=config['model_output_dim']).to(device)

	# Whip up a text dataset:
	text_dataset = RandomTextDataset(dataset_length=10000, example_length=config['model_output_characters'])
	
	# Random image dataset:
	#image_dataset = [i[0] for i in FakeData(size=1000, image_size=(3, 224, 224),)] + [Image.new("RGB", (224, 224)) for _ in range(100)]
	#image_dataset = [Image.new("RGB", (224, 224)) for _ in range(2)]
	image_dataset = ImageDataset("/home/joseph/MLData/train_512/*.jpg")

	# New meta-dataset:
	dataset = TextOverlayDataset(
		image_dataset = image_dataset, 
		text_dataset = text_dataset, 
		font_directory="./fonts/",
		font_sizes=[12, 16, 24, 48, 64, 96],
		#randomly_choose="image",
		maximum_font_translation_percent=0.2,
		maximum_font_rotation_percent=0.4,
		maximum_font_blur=3.2,
		long_text_behavior = 'empty',
		prefer_larger_fonts=True,
		pre_composite_transforms=[
			PILToTensor(),
			RandomResizedCrop(size=(224, 224), antialias=True),
			ToPILImage(),
		],
	)

	# Train:
	try:
		train(device, model, dataset, batch_size=config['batch_size'], num_epochs=config['num_epochs'], learning_rate=config['learning_rate'])
		torch.save(model, "./final_ocr_model.pt")
	except Exception as e:
		print(f"EXCEPTION: {e}")
		torch.save(model, "./ocrvision_model.pt")
		breakpoint()
		raise
	

if __name__=="__main__":
	main()
		


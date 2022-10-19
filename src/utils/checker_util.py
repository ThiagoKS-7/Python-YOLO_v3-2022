from src.utils.train_weights import Trainer
import os
weights_size = 248007048 # yolo.weights size (242.195kb)

class Weight_checker(object):
	def start(folder):
		size = 0
		for path, dirs, files in os.walk(folder):
			for f in files:
				fp = os.path.join(path, f)
				size += os.path.getsize(fp)

		if size == weights_size: 
			print("Loading weights...")
			Trainer.train()
			print("Done!")
		else:
			print("Weights already found!")
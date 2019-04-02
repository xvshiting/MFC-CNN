import numpy
from PIL import Image
import binascii
import os
import sys
import random   
import numpy as np
import CNN_Model
import creat_data
import load_data
import binary_2_image
# binary_2_image.image()
# creat_data.creat_data()
def train():
	binary_2_image.create_image()
	creat_data.creat_data()
	CNN_Model.train()
def retrain():
	CNN_Model.train()
def train_With_Aug():
	binary_2_image.create_image(big_size=40,aug_data=true)
	creat_data.creat_data()
	CNN_Model.train()

	


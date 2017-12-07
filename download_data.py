#!/usr/bin/python

# Imports
from urllib import request
import sys
import os

# locations
url = 'https://drive.google.com/uc?authuser=0&id=1c1UsjRnGe9tNAiJV6vRnzNLpGJ0t_bYl&export=download'
save_location = './data/forum-data.json'

# creating the folder for the data
if not os.path.exists('./data'):
	os.makedirs('./data')

# Sending the request
print("Downloading the data. This may take a little bit of time...")
sys.stdout.flush()
request.urlretrieve(url, save_location)
print("Download complete and saved to {}".format(save_location))
sys.stdout.flush()
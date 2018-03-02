#TODO preprocess iptv data to get user-channel view matrix
#1, how to transform the start time, end time into view matrix? 
# Probably slot time and use some threshold to binarize the view history 
import os
import shutil
import sys

import numpy as np
from scipy import sparse

import matplotlib.pyplot as plt
%matplotlib inline

import seaborn as sn
sn.set()

import pandas as pd

import tensorflow as tf
from tensorflow.contrib.layers import apply_regularization, l2_regularizer

import bottleneck as bn


### change `DATA_DIR` to the location where movielens-20m dataset sits
DATA_DIR = '/home/ubuntu/data/ml-20m/'

raw_data = pd.read_csv(os.path.join(DATA_DIR, 'ratings.csv'), header=0)


# binarize the data (only keep ratings >= 4)
raw_data = raw_data[raw_data['rating'] > 3.5]

raw_data.head()


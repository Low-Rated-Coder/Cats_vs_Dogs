import cv2
import os
import numpy as np
from random import shuffle
from tqdm import tqdm

TRAIN_DIR = '/home/vishal/Downloads/train'
TEST_DIR = '/home/vishal/Downloads/test'
IMG_SIZE = 50

# Defining label of image...
def label_img(img):
	image_name = img.split('.')[0]
	if image_name == 'cat':
		return [1,0]
	elif image_name == 'dog':
		return [0,1]

# Preprocessing train data....
def create_train_data():
	training_data = []
	for img in tqdm(os.listdir(TRAIN_DIR)):
		label = label_img(img)
		path = os.path.join(TRAIN_DIR,img)
		img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
		img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))
		training_data.append([np.array(img),np.array(label)])
	shuffle(training_data)
	np.save('train_data.npy',training_data)
	return training_data

# Preprocessing test data....
def create_test_data():
	testing_data = []
	for img in tqdm(os.listdir(TEST_DIR)):
		label = img.split('.')[0]
		path = os.path.join(TEST_DIR,img)
		img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
		img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))
		testing_data.append([np.array(img),label])
	shuffle(testing_data)
	np.save('test_data.npy',testing_data)
	return testing_data

# if already saved the data....
# train_data = np.load('train_data.npy')
# otherwise
# train_data = create_train_data()
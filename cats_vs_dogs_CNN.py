import tflearn
import os
import numpy as np
from tflearn.layers.conv import conv_2d,max_pool_2d
from tflearn.layers.core import input_data,fully_connected,dropout
from tflearn.layers.estimator import regression

from cats_vs_dogs_preprocessing import create_train_data,create_test_data

IMG_SIZE = 50
LEARNING_RATE = 0.001
MODEL_NAME = 'catsvsdogs-{}-{}.model'.format(LEARNING_RATE,'2conv-basic')

# Input layer
convnet = input_data(shape = [None,IMG_SIZE,IMG_SIZE,1],name = 'input')

# First Hidden Layer
convnet = conv_2d(convnet, 32, 5, activation = 'relu')
convnet = max_pool_2d(convnet, 5)

# Second Hidden Layer
convnet = conv_2d(convnet, 64, 5 ,activation = 'relu')
convnet = max_pool_2d(convnet, 5)

# Third Hidden Layer
convnet = conv_2d(convnet, 128, 5, activation = 'relu')
convnet = max_pool_2d(convnet, 5)

# Fourth Hidden Layer
convnet = conv_2d(convnet, 64, 5 ,activation = 'relu')
convnet = max_pool_2d(convnet, 5)

# Fifth Hidden Layer
convnet = conv_2d(convnet, 32, 5, activation = 'relu')
convnet = max_pool_2d(convnet, 5)

# Fully Connected Layer
convnet = fully_connected(convnet, 1024, activation = 'relu')
convnet = dropout(convnet, 0.8)

# Output Layer
convnet = fully_connected(convnet, 2, activation = 'softmax')
convnet = regression(convnet, optimizer = 'adam', learning_rate = LEARNING_RATE, loss = 'categorical_crossentropy', name = 'targets')

model = tflearn.DNN(convnet, tensorboard_dir = 'log')



# Check if model already exists....
if os.path.exists('{}.meta'.format(MODEL_NAME)):
	model.load(MODEL_NAME)
	print('Model Loaded !!')

# Creating train data...
train_data = create_train_data()

train = train_data[:-500]
test = train_data[-500:]

train_x = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
train_y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
test_y = [i[1] for i in test]

model.fit({'input':train_x}, {'targets':train_y}, n_epoch = 10, validation_set = ({'input':test_x}, {'targets':test_y}),
	snapshot_step = 500, show_metric = True, run_id = MODEL_NAME)

model.save(MODEL_NAME)
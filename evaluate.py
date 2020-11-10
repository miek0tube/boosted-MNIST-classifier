import numpy as np
import tensorflow as tf
import cv2
from utils import *
MAGIC = 424242
np.random.seed(MAGIC)
tf.set_random_seed(MAGIC)
import random
random.seed(MAGIC)

from tqdm import tqdm
from matplotlib import pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten  
from keras.layers import Convolution2D, MaxPooling2D 
from keras.utils import np_utils
from keras.datasets import mnist
from keras.datasets import mnist
 

(X_train, y_train), (X_test, y_test) = mnist.load_data() 

X_train = X_train.reshape(X_train.shape[0],28, 28, 1) 
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(y_train, 10) 
Y_test = np_utils.to_categorical(y_test, 10)

model = load_model("./mnist.h5")

SCORE_THRESHOLD = 0.5

eval_score_train = model.evaluate(X_train,Y_train)
eval_score_test = model.evaluate(X_test,Y_test)
print("Eval train score: " + str(eval_score_train))
print("Eval test score: " + str(eval_score_test))

score_train = get_score(model, X_train, Y_train)
print("My train score: " + str(score_train))
score_test = get_score(model, X_test, Y_test)
print("My test score: " + str(score_test))

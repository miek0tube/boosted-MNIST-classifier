import time
import numpy as np
import cv2
MAGIC = 424242

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.set_random_seed(MAGIC)

np.random.seed(MAGIC)
import random
random.seed(MAGIC)
import os
os.environ["PYTHONHASHSEED"]=str(MAGIC)
os.environ["TF_CUDNN_DETERMINISTIC"]="1"
os.environ["DETERMINISTIC_OPS"]="1"
os.environ["TF_CUDNN_USE_AUTOTUNE"]="0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="false"

import keras
from keras.utils import np_utils
from keras.datasets import mnist
from utils import *

from keras import backend as K
config = tf.ConfigProto()
config.gpu_options.allow_growth=False
config.gpu_options.allocator_type="BFC"# ""
config.gpu_options.force_gpu_compatible=True# false
#config.gpu_options.per_process_gpu_memory_fraction = 3.0# 1.0
config.graph_options.optimizer_options.global_jit_level=config.graph_options.optimizer_options.ON_2# 0
config.experimental.collective_deterministic_sequential_execution = True
sess = tf.Session(config=config)
K.set_session(sess)

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0],28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

model = create_classifier()

start = time.time()
train_model(model, X_train, Y_train)
end = time.time()
hours, rem = divmod(end-start, 3600)
minutes, seconds = divmod(rem, 60)
print("Training time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

model.save("./models/mnist"+str(score)+".h5")

import evaluate
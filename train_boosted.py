import time
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import numpy as np
MAGIC = 424242

tf.set_random_seed(MAGIC)

np.random.seed(MAGIC)
import random
random.seed(MAGIC)
import os
import cv2

os.environ["PYTHONHASHSEED"]=str(MAGIC)
os.environ["TF_CUDNN_DETERMINISTIC"]="1"
os.environ["DETERMINISTIC_OPS"]="1"
os.environ["TF_CUDNN_USE_AUTOTUNE"]="0"
os.environ["TF_FORCE_GPY_ALLOW_GROWTH"]="true"

import keras
from keras.utils import np_utils
from keras.datasets import mnist
from utils import *

from keras import backend as K
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
#config.gpu_options.allocator_type="BFC"# ""
#config.gpu_options.force_gpu_compatible=True# false
#config.gpu_options.per_process_gpu_memory_fraction = 3.0# 1.0
#config.graph_options.optimizer_options.global_jit_level=config.graph_options.optimizer_options.ON_2# 0
config.Experimental.collective_deterministic_sequential_execution = True
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

CLASSES_COUNT = 10
COUNT_OF_SAMPLES_PER_CLASS = 10
train_set_x = []
train_set_y = []

train_set_x, train_set_y = get_n_new_samples(  X_train, Y_train, CLASSES_COUNT, 3*COUNT_OF_SAMPLES_PER_CLASS )

step_index = 1
best_score = 0
patience = 3
bad_score_iterations = 0
best_weights = None

start = time.time()
while(len(train_set_x) < len(X_train) - COUNT_OF_SAMPLES_PER_CLASS*CLASSES_COUNT):
    print("-"*50)
    print("Iteration: %i, train data size: %i" % (step_index, len(train_set_x)) )
    train_model(model, train_set_x, train_set_y)

    scores = get_scores(model, X_train, Y_train)
    score = np.sum(scores) / len(scores)

    if (score < best_score):
        model.set_weights( best_weights )
        print( "-Accuracy didn't improve from %f" % best_score )
        bad_score_iterations += 1
        if bad_score_iterations > patience:
            model.set_weights( best_weights )
            print("Limit of %i bad iterations exceeded, stopping" % (patience))
            break
    else:
        print( "+Accuracy improved to %f from %f (%f pct)" % (score, best_score, 100.0 * (1 if best_score == 0 else score / best_score ) - 100.0) )
        best_score = score
        best_weights = model.get_weights()
        model.save("./models/mnist_[%i].[%i of %i].%f.h5" % (step_index, len(train_set_x), len(X_train), score))

    bad_score_iterations = 0

    train_set_x, train_set_y = add_n_worst_items(train_set_x, train_set_y, X_train, Y_train, scores, COUNT_OF_SAMPLES_PER_CLASS)

    step_index += 1

end = time.time()
hours, rem = divmod(end-start, 3600)
minutes, seconds = divmod(rem, 60)
print("Training time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

model.save("./models/mnist"+str(score)+".h5")

import test2
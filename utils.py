import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from tqdm import tqdm
import numpy as np
from PerforatorScheduler import *
import cv2

EPOCHS = 1000
BATCH_SIZE = 512*16
LR_START = 3e-4
TRAIN_TIMES = 5#3

def sort_class_data(a,b):
    return a[1]>b[1]

def add_n_worst_items(train_set_x, train_set_y, X_train, Y_train, scores, N):
    class_data={}

    for X, Y, score in zip(X_train, Y_train, scores):
        Y_class = np.argmax( Y )
        if Y_class in class_data:
            class_data[Y_class] += [[X, score]]
        else:
            class_data[Y_class] = [[X, score]]

    for class_elem in class_data:
        item = class_data[class_elem]
        item_sorted = sorted(item, key = lambda x: x[1] )
        item_selected = item_sorted[:N]

        for item in item_selected:
            train_set_x = np.resize(train_set_x, (train_set_x.shape[0] + 1, train_set_x.shape[1], train_set_x.shape[2], train_set_x.shape[3]))
            train_set_x[-1] = item[0]
            train_set_y = np.resize(train_set_y, (train_set_y.shape[0] + 1, train_set_y.shape[1]))

            class_matrix = np.zeros((train_set_y.shape[1]),dtype=np.float)
            class_matrix[class_elem] = 1.0
            train_set_y[-1] = class_matrix

    return train_set_x, train_set_y


def get_classes_count(y):
    classes_count={}
    for item in y:
        elem_class = np.argmax( item )
        if elem_class in classes_count:
            classes_count[elem_class] += 1
        else:
            classes_count[elem_class] = 1
    return classes_count

def get_n_new_samples(old_x, old_y, CLASSES_COUNT, COUNT_OF_SAMPLES_PER_CLASS):
    new_set_classes_y = []
    new_set_indexes = []

    items_to_add = CLASSES_COUNT * COUNT_OF_SAMPLES_PER_CLASS
    class_count_to_add = np.full((CLASSES_COUNT), COUNT_OF_SAMPLES_PER_CLASS)

    while(items_to_add > 0):
        items_added = 0
        for i in range(len(old_y)):
            elem_class = np.argmax( old_y[i] )
            if not (elem_class in new_set_classes_y) and (class_count_to_add[elem_class] > 0):
                items_added += 1

                class_count_to_add[elem_class] -= 1

                new_set_indexes.extend( [i])

        if items_added == 0:
            break

    new_set_x = np.empty((items_to_add, old_x.shape[1], old_x.shape[2], old_x.shape[3]),dtype=np.float32)
    new_set_y = np.empty((items_to_add),dtype=np.float32)

    for i in range(items_to_add):
        new_set_x[i] = old_x[new_set_indexes[i]]
        new_set_y[i] = np.argmax(old_y[new_set_indexes[i]])

    new_set_y = np_utils.to_categorical( new_set_y, 10 )

    return new_set_x, new_set_y

def get_batch_scores(model, batch, y_values, count, SCORE_THRESHOLD):
    scores = []
    predicted = model.predict( batch )
    for i in range(count):
        scores.append( iou( predicted[i], y_values[ i ], SCORE_THRESHOLD ) )

    return scores

def get_scores(model, x_values, y_values, SCORE_THRESHOLD = 0.5, BATCH_SIZE = 1024):
    scores = []
    batch_y = []
    batch_x = np.empty((BATCH_SIZE, x_values.shape[1], x_values.shape[2], 1), dtype=np.float32)
    batch_index = 0
    for i in tqdm(range(len(x_values))):
        batch_x[ batch_index ] = x_values[ i ]
        batch_y.append( y_values[ i ] )
        batch_index += 1

        if batch_index == BATCH_SIZE:
            scores.extend(get_batch_scores(model, batch_x, batch_y, batch_index, SCORE_THRESHOLD))
            batch_index = 0
            batch_y = []

    if batch_index > 0:
        scores.extend( get_batch_scores( model, batch_x, batch_y, batch_index, SCORE_THRESHOLD ) )
        del batch_x
        del batch_y

    return scores

def get_score(model, x_values, y_values, SCORE_THRESHOLD = 0.5):
    scores = get_scores(model, x_values, y_values, SCORE_THRESHOLD)

    return np.sum(scores) / len(scores)

def train_model(model, X_train, Y_train):
    callbacks=[]
    callbacks.append(
    tf.keras.callbacks.EarlyStopping(
        monitor="accuracy",
        min_delta=1e-3,
        patience=5,
        verbose=0,
        mode="auto",
        baseline=None,
        restore_best_weights=True,
    ))

    callbacks.append( PerforatorScheduler( verbose=True,
                                           monitor='loss',
                                           mode='min',
                                           factor=0.5,
                                           lr=LR_START,
                                           min_delta_fraction=1e-2,
                                           steps_per_epoch=TRAIN_TIMES * len(X_train)))

    model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=0, callbacks=callbacks)

    model.name=("MNIST epochs:%i batch:%i lr:%f" % (EPOCHS, BATCH_SIZE, LR_START))


def create_classifier():
    model = Sequential()

    model.add(Convolution2D(32, (3, 3), activation='elu', input_shape=(28,28,1)))
    model.add(Convolution2D(32, (3, 3), activation='elu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024, activation='elu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    optimizer = keras.optimizers.Adam(learning_rate = LR_START, amsgrad=True)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model

def iou(y_predicted, y_true, threshold=0.5):
    true_class = np.argmax(y_true)
    predicted_class = np.argmax(y_predicted)

    rating = 0
    rating_range = 1.0

    if true_class == predicted_class:
        rating = threshold
        rating_range = 1.0 - threshold

    _sum = np.sum(y_predicted)

    y_predicted /= _sum

    rating += y_predicted[true_class] * rating_range

    return rating

import sys
import numpy as np
import keras
import math
from keras import callbacks
import keras.backend as K

def softstep_sqrt(x):
    return math.pow(x, 0.5)

def softstep_sqr(x):
    return math.pow(x, 2.0)

def softstep_pow_sigm(x):
    x = (max(0.0, min( 1.0, x)) * 2.0) - 1.0
    pwr = math.pow(x, 0.25 ) if x > 0 else -math.pow(abs(x), 0.25)
    return ( pwr * 0.5) + 0.5

def softstep_sin(x):
    x = ((max(0.0, min( 1.0, x)) * 2.0) - 1.0) * (math.pi / 2)
    return ( math.sin(x) * 0.5) + 0.5

class PerforatorScheduler(keras.callbacks.Callback):
    def __init__(self, monitor='val_loss', factor=0.5,
                 verbose=False, mode='auto', min_delta_fraction=1e-2, steps_per_epoch=100, lr=1e-3,
                 **kwargs):
        super(PerforatorScheduler, self).__init__()

        self.monitor = monitor
        self.mode = mode
        self.min_delta_fraction = min_delta_fraction

        self.factor = factor
        if factor >= 1.0:
            raise ValueError('Not supported factor: ' + str(factor) + ' >= 1.0.')
        self.verbose = verbose
        self.min_lr = sys.float_info.min
        self.lr = self.min_lr
        self.target_lr = lr
        self.steps_per_epoch = steps_per_epoch
        self.wait = self.batch_count = self.epoch_start = self.best = 0
        self.monitor_op = None
        self.patience = 1
        self.best_weights = None
        self._reset()

    def _reset(self):
        """Resets wait counter and cooldown counter.
        """
        if self.mode not in ['auto', 'min', 'max']:
            print('Mode %s is unknown, fallback to auto mode.' % (self.mode),RuntimeWarning)
            self.mode = 'auto'
        if (self.mode == 'min' or
           (self.mode == 'auto' and 'acc' not in self.monitor)):
            self.monitor_op = lambda a, b: np.less(a, b*(1.0 - self.min_delta_fraction))
            self.best = np.Inf
        else:
            self.monitor_op = lambda a, b: np.greater(a, b*(1.0 + self.min_delta_fraction))
            self.best = -np.Inf
        self.wait = self.batch_count = 0
        self.patience = 1
        self.best_weights=None

    def on_train_begin(self, logs=None):
        self._reset()

        K.set_value(self.model.optimizer.lr, self.min_lr)

    def on_batch_end(self, batch, logs={}):
        self.batch_count = batch
        if (self.lr != self.target_lr):
            fraction = (batch - self.epoch_start + 1) / self.steps_per_epoch
            self.lr = self.lr_at_epoch_start + (self.target_lr - self.lr_at_epoch_start) * softstep_sin(fraction)
            K.set_value(self.model.optimizer.lr, self.lr)

        logs = logs or {}
        logs['lr'] = self.lr

    def on_epoch_begin(self, epoch, logs=None):
        self.lr_at_epoch_start = self.min_lr if epoch == 0 else self.lr

        self.epoch_start = self.batch_count

        if (self.verbose):
            if (abs(self.lr_at_epoch_start - self.target_lr) >= sys.float_info.min):
                print("Epoch: %i, LR will change from %f to %f [%i / %i]" % (epoch, self.lr_at_epoch_start, self.target_lr,self.wait,self.patience))
            else:
                print("Epoch: %i, LR will be stable at %f [[%i / %i]]"% (epoch, self.lr_at_epoch_start,self.wait,self.patience))

    def on_epoch_end(self, epoch, logs=None):
        self.lr = self.target_lr
        current = logs.get(self.monitor)
        if self.verbose:
            print("Current metric(%s): %f" % (self.monitor,current))
        if current is None:
            print('Metric `%s` is not available. Available metrics are: %s' %
                (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning)

        else:
            if self.monitor_op(current, self.best):
                self.best = current
                self.wait = 0
                self.patience = max( 1, self.patience // 2 - 1)
                self.best_weights = self.model.get_weights()
            else:
                self.wait += 1
                if self.wait > self.patience:
                    self.model.set_weights(self.best_weights)
                    if (self.verbose):
                        print("Stall detected, restarting")

                    self.patience += 1
                    old_lr = self.lr
                    new_lr = old_lr * self.factor

                    self.target_lr = new_lr
                    self.wait = 0

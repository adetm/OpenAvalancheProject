# Copyright (c) Microsoft Corporation
# All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
# to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import argparse
import logging
import os

import pandas as pd
import numpy as np

import tensorflow as tf
import keras
from keras.callbacks import EarlyStopping, TensorBoard
from keras.datasets import cifar10
from keras.optimizers import SGD, Adadelta, Adagrad, Adam, Adamax, RMSprop
from keras.utils import multi_gpu_model, to_categorical, Sequence
import keras.backend.tensorflow_backend as KTF

import nni
from nni.networkmorphism_tuner.graph import json_to_graph

data_directory = '/mnt/data/'

class AvySequence(Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.y_train_low = y_set[y_set['o_Day1DangerAboveTreeline']==0]
        self.y_train_moderate = y_set[y_set['o_Day1DangerAboveTreeline']==1]
        self.y_train_considerable = y_set[y_set['o_Day1DangerAboveTreeline']==2]
        self.y_train_high = y_set[y_set['o_Day1DangerAboveTreeline']==3]
        self.amount_per_class = int(batch_size/4)
        
    def __len__(self):
        return int(len(self.y_train_moderate)/(self.batch_size/self.amount_per_class))
    
    def __getitem__(self, idx):
        start = idx*self.amount_per_class
        y_train_batch = pd.concat([self.y_train_low.sample(n=self.amount_per_class, random_state=idx), 
                                   #self.y_train_moderate[start:start+self.amount_per_class],    
                                   self.y_train_moderate.sample(n=self.amount_per_class, random_state=idx),  
                                   self.y_train_considerable.sample(n=self.amount_per_class, random_state=idx), 
                                   self.y_train_high.sample(n=self.amount_per_class, random_state=idx)]).sample(frac=1, random_state=idx) #sample causes the rows to shuffle
        y_train_batch_cat = to_categorical(y_train_batch[['o_Day1DangerAboveTreeline', 'o_Day1DangerNearTreeline','o_Day1DangerBelowTreeline']])
        X_train_batch = self.x[y_train_batch.index.values]
        return X_train_batch, y_train_batch[:,0,:].copy()  #copy required to make C_Contiguous values

# set the logger format
log_format = "%(asctime)s %(message)s"
logging.basicConfig(
    filename="networkmorphism.log",
    filemode="a",
    level=logging.INFO,
    format=log_format,
    datefmt="%m/%d %I:%M:%S %p",
)
# set the logger format
logger = logging.getLogger("OAP-network-morphism-keras")


# restrict gpu usage background
config = tf.ConfigProto()
# pylint: disable=E1101,W0603
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

KTF.set_session(sess)


def get_args():
    """ get args from command line
    """
    parser = argparse.ArgumentParser("OAP")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--optimizer", type=str, default="SGD", help="optimizer")
    parser.add_argument("--epochs", type=int, default=200, help="epoch limit")
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="learning rate"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-5,
        help="weight decay of the learning rate",
    )
    return parser.parse_args()


trainloader = None
testloader = None
net = None
args = get_args()
TENSORBOARD_DIR = os.environ["NNI_OUTPUT_DIR"]


def build_graph_from_json(ir_model_json):
    """build model from json representation
    """
    graph = json_to_graph(ir_model_json)
    logging.debug(graph.operation_history)
    model = graph.produce_keras_model()
    return model


def parse_rev_args(receive_msg):
    """ parse reveive msgs to global variable
    """
    global trainloader
    global testloader
    global net

    # Loading Data
    logger.debug("Preparing data..")

    x_train = np.load(data_directory + '/X_train_noextrafeatures.npy')
    y_train = pd.read_csv(data_directory + '/y_train_noextrafeatures.csv', parse_dates=True)

    x_test = np.load(data_directory + '/X_test_noextrafeatures.npy')
    y_test = pd.read_csv(data_directory + '/y_test_noextrafeatures.csv', parse_dates=True)

    y_test = y_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    synthetic_regions = ['Low West', 'Low East', 'In The Desert', 'In The Lake']
    x_test = x_test[~y_test['UnifiedRegion'].isin(synthetic_regions)]
    x_train = x_train[~y_train['UnifiedRegion'].isin(synthetic_regions)]
    y_test = y_test[~y_test['UnifiedRegion'].isin(synthetic_regions)]
    y_train = y_train[~y_train['UnifiedRegion'].isin(synthetic_regions)]
    y_test = y_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    
    #y_train_cat = to_categorical(y_train[['o_Day1DangerAboveTreeline', 'o_Day1DangerNearTreeline','o_Day1DangerBelowTreeline']])
    y_test_cat = to_categorical(y_test[['o_Day1DangerAboveTreeline', 'o_Day1DangerNearTreeline','o_Day1DangerBelowTreeline']])
    
    trainloader = (x_train, y_train)
    testloader = (x_test, y_test_cat)

    # Model
    logger.debug("Building model..")
    net = build_graph_from_json(receive_msg)

    # parallel model
    try:
        available_devices = os.environ["CUDA_VISIBLE_DEVICES"]
        gpus = len(available_devices.split(","))
        if gpus > 1:
            net = multi_gpu_model(net, gpus)
    except KeyError:
        logger.debug("parallel model not support in this config settings")

    if args.optimizer == "SGD":
        optimizer = SGD(lr=args.learning_rate, momentum=0.9, decay=args.weight_decay)
    if args.optimizer == "Adadelta":
        optimizer = Adadelta(lr=args.learning_rate, decay=args.weight_decay)
    if args.optimizer == "Adagrad":
        optimizer = Adagrad(lr=args.learning_rate, decay=args.weight_decay)
    if args.optimizer == "Adam":
        optimizer = Adam(lr=args.learning_rate, decay=args.weight_decay)
    if args.optimizer == "Adamax":
        optimizer = Adamax(lr=args.learning_rate, decay=args.weight_decay)
    if args.optimizer == "RMSprop":
        optimizer = RMSprop(lr=args.learning_rate, decay=args.weight_decay)

    # Compile the model
    net.compile(
        loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
    )
    return 0


class SendMetrics(keras.callbacks.Callback):
    """
    Keras callback to send metrics to NNI framework
    """

    def on_epoch_end(self, epoch, logs=None):
        """
        Run on end of each epoch
        """
        if logs is None:
            logs = dict()
        logger.debug(logs)
        nni.report_intermediate_result(logs["val_acc"])




# Training
def train_eval():
    """ train and eval the model
    """

    global trainloader
    global testloader
    global net

    seq = AvySequence(trainloader[0], trainloader[1], args.batch_size)
    #(x_train, y_train) = trainloader
    (x_test, y_test) = testloader

    # train procedure
    net.fit_generator(
        generator=seq,
        validation_data=(x_test, y_test[:,0,:]),
        epochs=args.epochs,
        shuffle=True,
        callbacks=[
            SendMetrics(),
            EarlyStopping(min_delta=0.001, patience=10),
            TensorBoard(log_dir=TENSORBOARD_DIR),
        ],
    )

    # trial report final acc to tuner
    _, acc = net.evaluate(x_test, y_test)
    logger.debug("Final result is: %.3f", acc)
    nni.report_final_result(acc)


if __name__ == "__main__":
    try:
        # trial get next parameter from network morphism tuner
        RCV_CONFIG = nni.get_next_parameter()
        logger.debug(RCV_CONFIG)
        parse_rev_args(RCV_CONFIG)
        train_eval()
    except Exception as exception:
        logger.exception(exception)
        raise

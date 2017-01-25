#!/usr/bin/env python

import sys
import time

import numpy as np

from keras.models import Graph
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.optimizers import Adagrad
from keras.callbacks import ModelCheckpoint, EarlyStopping

np.random.seed(0)

FILTER_LEN1 = 10
FILTER_LEN2 = 20
NB_FILTER1 = 1000
NB_FILTER2 = 1000
NB_HIDDEN = 2000
POOL_FACTOR = 1
DROP_OUT_CNN = 0.1
DROP_OUT_MLP = 0.1
ACTIVATION = 'relu'
BATCH_SIZE = 512/4
NB_EPOCH = 100
LR = 0.01/2


def main():
    save_name = sys.argv[1]
    nb_filter1 = int(sys.argv[2])
    nb_filter2 = int(sys.argv[3])
    nb_hidden = int(sys.argv[4])
    dropout_cnn = float(sys.argv[5])
    dropout_mlp = float(sys.argv[6])
    filter_len1 = int(sys.argv[7])
    filter_len2 = int(sys.argv[8])
    
    print 'loading data...'
    sys.stdout.flush()

    X_tr = np.load('X_tr_float32.npy')
    Y_tr = np.load('Y_tr_float32.npy')
    X_va = np.load('X_va_float32.npy')
    Y_va = np.load('Y_va_float32.npy')
    X_te = np.load('X_te_float32.npy')
    Y_te = np.load('Y_te_float32.npy')

    __, seq_len, channel_num = X_tr.shape
    pool_len1 = (seq_len-filter_len1+1)/POOL_FACTOR
    pool_len2 = (seq_len-filter_len2+1)/POOL_FACTOR
    
    model = Graph()
    
    model.add_input(name='input', input_shape=(seq_len, channel_num))
    
    #convolution layer 1
    model.add_node(Convolution1D(input_dim=channel_num,
                        input_length=seq_len,
                        nb_filter=nb_filter1,
                        border_mode='valid',
                        filter_length=filter_len1,
                        activation=ACTIVATION),
                   name='conv1', input='input')
    model.add_node(MaxPooling1D(pool_length=pool_len1, stride=pool_len1), name='maxpool1', input='conv1')
    model.add_node(Dropout(dropout_cnn), name='drop_cnn1', input='maxpool1')
    model.add_node(Flatten(), name='flat1', input='drop_cnn1')


    #convolution layer 2
    model.add_node(Convolution1D(input_dim=channel_num,
                        input_length=seq_len,
                        nb_filter=nb_filter2,
                        border_mode='valid',
                        filter_length=filter_len2,
                        activation=ACTIVATION),
                   name='conv2', input='input')
    model.add_node(MaxPooling1D(pool_length=pool_len2, stride=pool_len2), name='maxpool2', input='conv2')
    model.add_node(Dropout(dropout_cnn), name='drop_cnn2', input='maxpool2')
    model.add_node(Flatten(), name='flat2', input='drop_cnn2')

    
    model.add_node(Dense(nb_hidden), name='dense1', inputs=['flat1', 'flat2'])
    model.add_node(Activation('relu'), name='act1', input='dense1')
    model.add_node(Dropout(dropout_mlp), name='drop_mlp1', input='act1')

    model.add_node(Dense(input_dim=nb_hidden, output_dim=1), name='dense2', input='drop_mlp1')
    model.add_node(Activation('sigmoid'), name='act2', input='dense2')

    model.add_output(name='output', input='act2')

    adagrad = Adagrad(lr=LR)
 
    print 'model compiling...'
    sys.stdout.flush()
     
    model.compile(loss={'output':'binary_crossentropy'}, optimizer=adagrad)
   
    checkpointer = ModelCheckpoint(filepath=save_name+'.hdf5', verbose=1, save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

    outmodel = open(save_name+'.json', 'w')
    outmodel.write(model.to_json())
    outmodel.close()
    
    print 'training...'
    sys.stdout.flush()
    
    time_start = time.time()
    model.fit({'input':X_tr, 'output':Y_tr}, batch_size=BATCH_SIZE, nb_epoch=NB_EPOCH, 
              verbose=1, validation_data={'input':X_va, 'output':Y_va},
              callbacks=[checkpointer, earlystopper])
    time_end = time.time()
    
    model.load_weights(save_name+'.hdf5')
    n_va = Y_va.shape[0]
    n_te = Y_te.shape[0]
    Y_va_hat = np.round(model.predict({'input':X_va}, BATCH_SIZE, verbose=1)['output'])
    Y_te_hat = np.round(model.predict({'input':X_te}, BATCH_SIZE, verbose=1)['output'])
#     loss_va = model.evaluate({'input':X_va, 'output':Y_va})
#     loss_te = model.evaluate({'input':X_te, 'output':Y_te})
    acc_va = 1-np.abs(Y_va-Y_va_hat).sum()/n_va
    acc_te = 1-np.abs(Y_te-Y_te_hat).sum()/n_te


    print '*'*100
    print '%s accuracy_va : %.4f' % (save_name, acc_va)
    print '%s accuracy_te : %.4f' % (save_name, acc_te)
    print '%s training time : %d sec' % (save_name, time_end-time_start)
    
if __name__ == '__main__':
    main()



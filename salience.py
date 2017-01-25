#!/usr/bin/env python

import sys
import os

import numpy as np
import theano as th
import theano.tensor as T
from keras.models import model_from_json


NT_DICT = {'1000':'A', '0100':'C', '0010':'G', '0001':'T', '0000':'N'}


def main():
    X = np.load(sys.argv[1])
    base_name = sys.argv[2]
    save_name = sys.argv[3]
    
    base_json = base_name+'.json'
    base_hdf5 = base_name+'.hdf5'
    model = model_from_json(open(base_json).read())
    model.load_weights(base_hdf5)
    
    N, seq_len, channel_num = X.shape
    
    x = model.get_input()
    a = model.get_output()
    y = T.log(a/(1-a))
    g = T.grad(T.mean(y), x)
    f_g = th.function([x], g)
    
    X_i = np.random.normal(0.25, 0.01, (2, seq_len, channel_num)).astype('float32')
    
    outfile = open(save_name, 'w')
    for i in range(0, N):
        X_i[0] = X[i]
        X_i[1] = X[i]
        salience_i = (f_g(X_i).mean(axis=0)*X[i]).sum(axis=1)
        
        seq = ''
        j = 0
        while j < seq_len:
            nt = NT_DICT[''.join(map(lambda x:str(int(x)), X[i, j, :]))]
            
            if nt == 'N':
                break
            
            seq += nt
            j += 1
            
        salience_str = ','.join(map(lambda x:'%.3f' %(x), salience_i[0:j]))
        
        outfile.write(seq + '\t' + salience_str + '\n')

        if (i+1)%1000 == 0:
            print '%s/%s...' % (i+1, N)
        
        sys.stdout.flush()
        
    outfile.close()
    
    
    
if __name__ == '__main__':
    main()



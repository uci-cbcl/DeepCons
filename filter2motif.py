#!/usr/bin/env python

import sys

import numpy as np
import theano
from keras.models import model_from_json

BATCH_SIZE = 512/4
MEME_HEADER = """MEME version 4.4

ALPHABET= ACGT

strands: + -

Background letter frequencies (from web form):
A 0.25000 C 0.25000 G 0.25000 T 0.25000 

"""



def update_counts(counts, n_sites, x, a):
    a_max = a.max(axis=1)
    a_max_idx = a.argmax(axis=1)
    
    n, seq_len, channel_num = x.shape
    nb_filter, filter_len, channel_num = counts.shape
    
    for i in range(0, n):
        for j in range(0, nb_filter):
            idx = a_max_idx[i, j]
            counts[j] += a_max[i, j]*x[i, idx:idx+filter_len, :]
            n_sites[j] += a_max[i, j]
    
    return (counts, n_sites)
    


def main():
    X = np.load(sys.argv[1])
    base_name = sys.argv[2]
    
    base_json = base_name+'.json'
    base_hdf5 = base_name+'.hdf5'
    base_meme = base_name+'.meme'
    
    model = model_from_json(open(base_json).read())
    model.load_weights(base_hdf5)
    
    N, seq_len, channel_num = X.shape
    
    _,  act_len1, nb_filter1 = model.nodes['conv1'].output_shape
    _,  act_len2, nb_filter2 = model.nodes['conv2'].output_shape
    nb_filter1, channel_num, filter_len1, _ = model.nodes['conv1'].get_weights()[0].shape
    nb_filter2, channel_num, filter_len2, _ = model.nodes['conv2'].get_weights()[0].shape
    f1 = theano.function([model.nodes['conv1'].get_input()], model.nodes['conv1'].get_output())
    f2 = theano.function([model.nodes['conv2'].get_input()], model.nodes['conv2'].get_output())
    
    counts1 = np.zeros((nb_filter1, filter_len1, channel_num))+1e-5
    counts2 = np.zeros((nb_filter2, filter_len2, channel_num))+1e-5
    n_sites1 = np.zeros(nb_filter1)
    n_sites2 = np.zeros(nb_filter2)
    
    i = 0
    while i+BATCH_SIZE < N:
        x = X[i:i+BATCH_SIZE]

        a1 = f1(x)
        a2 = f2(x)
        counts1, n_sites1 = update_counts(counts1, n_sites1, x, a1)
        counts2, n_sites2 = update_counts(counts2, n_sites2, x, a2)
        
        i += BATCH_SIZE
        
        print '%s/%s data points processed...' % (i, N)
        sys.stdout.flush()
        
    
    x = X[i:N]
    a1 = f1(x)
    a2 = f2(x)
    counts1, n_sites1 = update_counts(counts1, n_sites1, x, a1)
    counts2, n_sites2 = update_counts(counts2, n_sites2, x, a2)
    pwm1 = counts1/counts1.sum(axis=2).reshape(nb_filter1, filter_len1, 1)
    pwm2 = counts2/counts2.sum(axis=2).reshape(nb_filter2, filter_len2, 1)
    
    outfile = open(base_meme, 'w')
    outfile.write(MEME_HEADER)
    
    for i in range(0, nb_filter1):
        outfile.write('MOTIF FILTER_LEN%s_%s\n\n' % (filter_len1, i))
        outfile.write('letter-probability matrix: alength= 4 w= %s nsites= %s E= 1e-6\n' % (filter_len1, int(n_sites1[i])))
        
        for j in range(0, filter_len1):
            outfile.write('%f\t%f\t%f\t%f\n' % tuple(pwm1[i, j, :].tolist()))
        
        outfile.write('\n')
        
    for i in range(0, nb_filter2):
        outfile.write('MOTIF FILTER_LEN%s_%s\n\n' % (filter_len2, i))
        outfile.write('letter-probability matrix: alength= 4 w= %s nsites= %s E= 1e-6\n' % (filter_len2, int(n_sites2[i])))
        
        for j in range(0, filter_len2):
            outfile.write('%f\t%f\t%f\t%f\n' % tuple(pwm2[i, j, :].tolist()))
        
        outfile.write('\n')

    
    
    
    
if __name__ == '__main__':
    main()
    
    



README for DeepCons
===================

INTRODUCTION
============
Comparative genomics has been very effective in finding functional elements across the human genome. However, understanding the functional roles of these sequences still remain a challenge, especially in noncoding regions. We present a deep learning approach, DeepCons, to understand sequence conservation. DeepCons is a convolutional neural network that is trained to classify conserved and non-conserved sequences. We show that the learned convolution kernels of DeepCons can capture rich information with respect to sequence conservation: 1) they match motifs such as CTCF, JUND, RFX3 and MEF2A that are known to be widely distributed within conserved noncoding elements, 2) they have positional bias relative to transcription start sites, transcription end sites and miRNA, and 3) they have strand bias relative to transcription end sites. DeepCons could also be used to score sequence conservation at nucleotide level resolution. We rediscovered known motifs within a given sequence by highlighting each nucleotide regarding their scores.


PREREQUISITES
=============
* Python (2.7). [Python 2.7.11](https://www.python.org/downloads/release/python-2711/) is recommended.

* [Numpy](http://www.numpy.org/)(>=1.10.4). 

* [Scipy](http://www.scipy.org/)(>=0.17.0). 

* [Theano](https://github.com/Theano/Theano/releases/tag/rel-0.8.2)(0.8.2).

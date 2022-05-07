'''
Imports
'''
!pip install uniseg

import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import unicodedata
import re
import numpy as np
import os
import io
import time
import random
import shutil
from matplotlib.font_manager import FontProperties
import shutil
#HTMl library to generate the connectivity html file
from IPython.display import HTML as html_print
from IPython.display import display
from attention import *
from best_train import *
from dataset_util import *
from decoder import *
from encoder import *
from hyperparameter_tuning_train import *
from prediction_plots_util import *
from train_util import *
'''
Class - GRU Encoder
'''
class GRU_Encoder(tf.keras.Model):
  #Initialization
  def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz, dropout=0):
    super(GRU_Encoder, self).__init__()
    self.batch_sz = batch_sz    #batch_size
    self.enc_units = enc_units  #encoder_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim) #embeding dimensions and layer initialization
    #keras GRU layer
    self.gru = tf.keras.layers.GRU(self.enc_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform',
                                   dropout = dropout)
  #calling the GRU encoder
  def call(self, x, hidden):
    #calling the embeding initializations
    x = self.embedding(x)
    #return the encoder output and the state
    output, state = self.gru(x, initial_state=hidden)
    return output, state

  #initialiaztion of the hidden states
  def initialize_hidden_state(self):
    return tf.zeros((self.batch_sz, self.enc_units))

'''
class - LSTM encoder
'''
class LSTM_Encoder(tf.keras.Model):
  #initialization function
  def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz,dropout=0):
    super(LSTM_Encoder, self).__init__()
    self.batch_sz = batch_sz    #batch_size
    self.enc_units = enc_units  #encoder_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim) #embedding_dimensions
    #keras LSTM layer
    self.lstm = tf.keras.layers.LSTM(self.enc_units, 
                         return_sequences=True, 
                         return_state=True,
                         recurrent_initializer='glorot_uniform',
                         dropout = dropout)

  #call function
  def call(self, x, hidden,cell_state):
    #embedding layer calling
    x = self.embedding(x)
    #output and the last cell calling
    output, last_hidden,last_cell_state = self.lstm(x, initial_state=[hidden,cell_state])
    #return the output, last hidden and the last cell state
    return output, last_hidden,last_cell_state
    
  #initialization of the hidden state
  def initialize_hidden_state(self):
      return tf.zeros((self.batch_sz, self.enc_units)), tf.zeros((self.batch_sz, self.enc_units))

'''
class - RNN encoder
'''
class RNN_Encoder(tf.keras.Model):
  #intialization function
  def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz,dropout=0):
    super(RNN_Encoder, self).__init__()
    self.batch_sz = batch_sz    #batch size
    self.enc_units = enc_units  #encoder units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)   #embedding dimensions
    #keras RNN layer
    self.rnn = tf.keras.layers.SimpleRNN(self.enc_units, 
                         return_sequences=True, 
                         return_state=True,
                         recurrent_initializer='glorot_uniform',
                         dropout = dropout)

  #call function
  def call(self, x, hidden):
    #embedding layer calling
    x = self.embedding(x)
    #returning the output and the final state
    output, final_state = self.rnn(x,initial_state=hidden)
    return output, final_state
    
  #initialization of the hidden states
  def initialize_hidden_state(self):
      return tf.zeros((self.batch_sz, self.enc_units))
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
class GRU Decoder 
'''
class GRU_Decoder(tf.keras.Model):
  #initialization
  def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz,dropout=0):
    super(GRU_Decoder, self).__init__()
    self.batch_sz = batch_sz    #batch size
    self.dec_units = dec_units  #decoder units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim) #embeding layer initialization
    #keras GRU layer
    self.gru = tf.keras.layers.GRU(self.dec_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform',dropout=dropout)
    #dense or fully connected layer
    self.fc = tf.keras.layers.Dense(vocab_size)

    #using the attention
    self.attention = BahdanauAttention(self.dec_units)

  #call function to generate the output, state and the attention weights
  def call(self, x, hidden, enc_output):
    # enc_output shape == (batch_size, max_length, hidden_size)
    context_vector, attention_weights = self.attention(hidden, enc_output)
    # output shape after passing through embedding == (batch_size, 1, embedding_dim)
    output = self.embedding(x)
    # output shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
    output = tf.concat([tf.expand_dims(context_vector, 1), output], axis=-1)
    # passing the concatenated vector to the GRU
    output, state = self.gru(output)
    # output shape == (batch_size * 1, hidden_size)
    output = tf.reshape(output, (-1, output.shape[2]))
    # output shape == (batch_size, vocab)
    output = self.fc(output)
    #return the output, state and the attention weights.
    return output, state, attention_weights

'''
class LSTM decoder
'''
class LSTM_Decoder(tf.keras.Model):
  #initialization
  def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz,dropout=0):
    super(LSTM_Decoder, self).__init__()
    self.batch_sz = batch_sz    #batch size
    self.dec_units = dec_units  #decoder units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim) #embedding
    #keras LSTM layer
    self.lstm = tf.keras.layers.LSTM(self.dec_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform',dropout=dropout)
    #dense/ Fully connected layer
    self.fc = tf.keras.layers.Dense(vocab_size)

    #applying the attention layer
    self.attention = BahdanauAttention(self.dec_units)

  #call function generating output, hiddden and cell state and the attention weights
  def call(self, x, hidden, enc_output,cell_state):
    # enc_output shape == (batch_size, max_length, hidden_size)
    context_vector, attention_weights = self.attention(hidden, enc_output)
    # x shape after passing through embedding == (batch_size, 1, embedding_dim)
    output = self.embedding(x)
    # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
    output = tf.concat([tf.expand_dims(context_vector, 1), output], axis=-1)
    # passing the concatenated vector to the GRU
    output, last_hidden_state,last_cell_state = self.lstm(output,initial_state=[hidden,cell_state])
    # output shape == (batch_size * 1, hidden_size)
    output = tf.reshape(output, (-1, output.shape[2]))
    # output shape == (batch_size, vocab)
    output = self.fc(output)
    #returning output, hiddden and cell state and the attention weights
    return output, [last_hidden_state,last_cell_state], attention_weights

'''
Class RNN decoder
'''
class RNN_Decoder(tf.keras.Model):
  #initialization
  def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz,dropout=0):
    super(RNN_Decoder, self).__init__()
    self.batch_sz = batch_sz    #batch size
    self.dec_units = dec_units  #decoder unnits
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim) #embedding layer
    #keras RNN layer
    self.rnn = tf.keras.layers.SimpleRNN(self.dec_units, 
                         return_sequences=True, 
                         return_state=True,
                         recurrent_initializer='glorot_uniform',
                         dropout = dropout)
    #dense/ fully connected layer
    self.fc = tf.keras.layers.Dense(vocab_size)
    #applying attention layer
    self.attention = BahdanauAttention(self.dec_units)


  #call function generating the output state and the attention weights
  def call(self, x, hidden, enc_output):
    # enc_output shape == (batch_size, max_length, hidden_size)
    context_vector, attention_weights = self.attention(hidden, enc_output)
    # x shape after passing through embedding == (batch_size, 1, embedding_dim)
    output = self.embedding(x)
    # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
    output = tf.concat([tf.expand_dims(context_vector, 1), output], axis=-1)
    # passing the concatenated vector to the GRU
    output, final_state = self.rnn(output)
    # output shape == (batch_size * 1, hidden_size)
    output = tf.reshape(output, (-1, output.shape[2]))
    # output shape == (batch_size, vocab)
    output = self.fc(output)
    #return the output state and the attention weights
    return output, final_state, attention_weights
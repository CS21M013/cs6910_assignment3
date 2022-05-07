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
Attention class (Bhadanau Attention) refernce for the attention  - https://arxiv.org/abs/1409.0473
'''
class BahdanauAttention(tf.keras.layers.Layer):
  #initialization
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)  #W_1
    self.W2 = tf.keras.layers.Dense(units)  #W_2
    self.V = tf.keras.layers.Dense(1)       #V

  '''
  call function genrating the context vector and the attention weights
  '''
  def call(self, query, values):
    '''
    shape of query hidden state == (batch_size, hidden size)
    shape of query_with_time_axis == (batch_size, 1, hidden size)
    shape of values  == (batch_size, max_len, hidden size)
    '''
    #To broadcast addition along the time axis to calculate the score
    query_with_time_axis = tf.expand_dims(query, 1)

    # shape of score == (batch_size, max_length, 1)
    # shape of the tensor before applying self.V is (batch_size, max_length, units)
    e = self.V(tf.nn.tanh(
        self.W1(query_with_time_axis) + self.W2(values)))

    # shape of attention_weights == (batch_size, max_length, 1)
    #generating the attention weights
    attn_wts = tf.nn.softmax(e, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    #generating the context vector
    context = attn_wts * values
    context = tf.reduce_sum(context, axis=1)

    #returning the context vector and the attention weights
    return context, attn_wts
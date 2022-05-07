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
Preprocessing the words
'''
def word_process(w):
  #adding the tab and next line characters in the words
  w = '\t' + w + '\n'
  return w

'''
Function - Returns pairs of target word,input word.
'''
def create_dataset(path):
  lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
  #creating the word pairs
  word_pairs = [[word_process(w)  for w in line.split('\t')[:-1]]
                for line in lines[:-1]]
  return zip(*word_pairs)

'''
Function - tokenize the language
'''
def tokenize(lang):
  #using keras text tokenizer
  lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', char_level=True)
  lang_tokenizer.fit_on_texts(lang)
  #generating the sequence
  tensor = lang_tokenizer.texts_to_sequences(lang)
  #tensor used for pading the sequences
  tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')
  #retun the tensor and the language tokenizer
  return tensor, lang_tokenizer

'''
Function - load_dataset 
'''
def load_dataset(path):
  #creating the target word and input word pairs
  output_lang, inp_lang = create_dataset(path)

  #generating the tokenized tensor for the input words
  input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
  #generating the tokenized tensor for the target words
  output_tensor, output_lang_tokenizer = tokenize(output_lang)

  return input_tensor, output_tensor, inp_lang_tokenizer, output_lang_tokenizer
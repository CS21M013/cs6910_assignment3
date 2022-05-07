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
Fucntion - Calculating the loss function
Reference: https://stackoverflow.com/questions/62916592/loss-function-for-sequences-in-tensorflow-2-0
'''
def calculate_loss(real, pred):
  mask_position = tf.math.logical_not(tf.math.equal(real, 0))
  loss_value = loss_object(real, pred)

  mask_position = tf.cast(mask_position, dtype=loss_value.dtype)
  loss_value *= mask_position

  #returns the mean of the loss value
  return tf.reduce_mean(loss_value)
  
'''
Function - train_batch
calculates the loss of the batch and returns the batch loss after training every batch in each epoch
'''
@tf.function
def train_batch(inp, targ, enc_hidden, enocder, decoder,rnn_type):
  #Final loss value
  loss = 0
  with tf.GradientTape() as tape:
        #checking if it is GRU or RNN
        if rnn_type!='LSTM':
            enc_output, enc_hidden = encoder(inp, enc_hidden)
            dec_hidden = enc_hidden
        #checking if it is LSTM
        elif rnn_type=='LSTM':
            enc_output, enc_hidden,enc_cell_state = encoder(inp, enc_hidden[0],enc_hidden[1])
            dec_hidden = enc_hidden
            dec_cell_state=enc_cell_state
        
        #geting the decoder input
        dec_input = tf.expand_dims([targ_lang.word_index['\t']] * BATCH_SIZE, 1)
        
        #Teacher forcing
        for t in range(1, targ.shape[1]):
            if rnn_type!='LSTM':
                # passing enc_output to the decoder if it is RNN or GRU
                predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
            elif rnn_type=='LSTM':
                if t==1:
                  # passing enc_output to the decoder if it is a LSTM
                  predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output,dec_cell_state)
                elif t>1:
                  # passing enc_output to the decoder if it is a LSTM
                  predictions, dec_hidden, _ = decoder(dec_input, dec_hidden[0], enc_output,dec_cell_state)
            #calculating the loss using calculate loss function
            loss += calculate_loss(targ[:, t], predictions)
            #using teacher forcing
            dec_input = tf.expand_dims(targ[:, t], 1)
  #calculating the batch loss
  batch_loss = (loss / int(targ.shape[1]))
  #calculate the variables
  variables = encoder.trainable_variables + decoder.trainable_variables
  #calculate the gradients
  gradients = tape.gradient(loss, variables)
  #applying the gradients to the optimizer
  optimizer.apply_gradients(zip(gradients, variables))
  return batch_loss
  
'''
Function - inference_model
generating the predicted word, input word, attention weights and the attention plot
'''
def inference_model(input_word,rnn_type):
  #creating an empty attention plot
  attention_plot = np.zeros((max_length_targ, max_length_inp))

  #preprocessing the input word
  input_word = preprocess_word(input_word)

  #converting the word to tensor after pading
  inputs = [inp_lang.word_index[i] for i in input_word]
  inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=max_length_inp, padding='post')
  inputs = tf.convert_to_tensor(inputs)

  #predicted word initialization
  predicted_word = ''
  
  #if cell type is GRU or RNN
  if rnn_type!='LSTM':
    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden = encoder(inputs, hidden)
    dec_hidden = enc_hidden
  #if cell type is LSTM
  elif rnn_type=='LSTM':
    hidden=tf.zeros((1, units))
    cell_state= tf.zeros((1, units)) 
    enc_out, enc_hidden,enc_cell_state = encoder(inputs, hidden,cell_state)
    dec_hidden = enc_hidden

  #generating the decode inputs
  dec_input = tf.expand_dims([targ_lang.word_index['\t']], 0)

  #storing the attention weights
  att_w=[]

  #calculating the predictions
  for t in range(max_length_targ):
    #if cell is GRU or RNN
    if rnn_type!='LSTM':
      predictions, dec_hidden, attention_weights = decoder(dec_input,dec_hidden,enc_out)
    #if cell is LSTM
    elif rnn_type=='LSTM':
      predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out, enc_cell_state)
      dec_hidden=dec_hidden[0]

    # storing the attention weights for plotting latter
    attention_weights = tf.reshape(attention_weights, (-1, ))
    attention_plot[t] = attention_weights.numpy()
    att_w.append(attention_weights.numpy()[0:len(input_word)])
    

    #predicted id
    predicted_id = tf.argmax(predictions[0]).numpy()
    #predicted word
    predicted_word += targ_lang.index_word[predicted_id] 

    #in case of last character
    if targ_lang.index_word[predicted_id] == '\n':
      return predicted_word, input_word, attention_plot,att_w

    # the predicted ID is fed back into the model
    dec_input = tf.expand_dims([predicted_id], 0)
  #finally return the predicted word, input word, attention plot and the attention weight
  return predicted_word, input_word, attention_plot,att_w
  
  
'''
Function - Validate
returning the validation or the testing accuracy on the validation data
'''
def validate(path_to_file,folder_name):
  #while testing to generate the predictions files (output files)
  save = False
  if path_to_file.find("test")!=-1:
    if os.path.exists(os.path.join(os.getcwd(),"predictions_attention",str(folder_name))):
      shutil.rmtree(os.path.join(os.getcwd(),"predictions_attention",str(folder_name)))
      
    if not os.path.exists(os.path.join(os.getcwd(),"predictions_attention")):
        os.mkdir(os.path.join(os.getcwd(),"predictions_attention"))
    os.mkdir(os.path.join(os.getcwd(),"predictions_attention",str(folder_name)))
    success_file = open(os.path.join(os.getcwd(),"predictions_attention",str(folder_name),"success.txt"),"w",encoding='utf-8', errors='ignore')
    failure_file = open(os.path.join(os.getcwd(),"predictions_attention",str(folder_name),"failure.txt"),"w",encoding='utf-8', errors='ignore')
    save=True
    
  #the count of the correct predictions
  success_count=0
  # Get the target words and input words for the validation
  target_words, input_words = create_dataset(path_to_file)
  for i in range(len(input_words)):
    #generate the predicted words for the corresponding input words
    predicted_word, input_word, attention_plot,att_w = inference_model(input_words[i],rnn_type)
    record= input_word.strip()+' '+target_words[i].strip()+' '+predicted_word[:-1].strip()+"\n"
    # The last character of target_words[i] and predicted word is '\n', first character of target_words[i] is '\t'
    if target_words[i][1:]==predicted_word:
      #increasing the accuracy count
      success_count = success_count + 1
      if save == True:
        success_file.write(record)
    elif save==True:
      failure_file.write(record)

  #saving the files
  if save==True:
    success_file.close()
    failure_file.close()
    
  #return the acuracy
  return success_count/len(input_words)
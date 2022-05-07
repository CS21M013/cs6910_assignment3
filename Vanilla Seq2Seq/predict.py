'''
Imports
'''
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import copy
'''
Function - do_predictions (Decoding the entire batch to generate the predictions)
Input - 
  input_seq
  encoder_model
  decoder_model
  batch-size
  encoder_layers
  decoder_layers
Output - 
  Predicted words
'''
def do_predictions(input_seq,encoder_model,decoder_model,batch_size,encoder_layers,decoder_layers):
    # use the encoder model to get the value of the states
    sv = encoder_model.predict(input_seq) #values of the states
    #if GRU or RNN
    if rnn_type=='GRU' or 'RNN':
      sv=[sv]
    #save states value for RNN, LSTM as well as GRU
    nl=sv

    #keep on adding the states value for every deocoder layer
    for i in range(decoder_layers-1):
      nl=nl+sv
    sv=nl
    
    #contains previously predicted character's index for every words in batch.
    prev_index = np.zeros((batch_size, 1))
    # starting with \t for every word in batch hence tokenize.
    prev_index[:, 0] = target_tokenizer.word_index['\t']
    
    #predicted words list
    word_predictions = [ "" for i in range(batch_size)]
    #check if batch predicted or not
    check=[False for i in range(batch_size)]

    for i in range(max_decoder_seq_length):
        out = decoder_model.predict(tuple([prev_index] + sv)) #predictions of the decoder model based on the previous char index
        out_prob=out[0] #Probability as a result of the softmax function
        sv = out[1:] #decoder states value is stored.
        #for every batch we execute the following
        for j in range(batch_size):
          #if bacth already done
          if check[j]:
            continue          
          
          sampled_char_index = np.argmax(out_prob[j, -1, :]) #geting the sample token index
          #if sampled index is 0 then character is nextline character
          if sampled_char_index == 0:
            sampled_char='\n'
          # otherwise convert index to the respective character
          else:
            sampled_char = index_to_char_target[sampled_char_index]
          #check if it is ending
          if sampled_char == '\n':
            check[j]=True
            continue
          #uGet the predicted words value       
          word_predictions[j] += sampled_char
          #update the previously predicted characters        
          prev_index[j,0]=target_tokenizer.word_index[sampled_char]
    #return the predicted words.
    return word_predictions
'''
Imports
'''
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import copy
from predict import do_predictions
'''
Function - test_accuracy (calculate the word level accuracy (Testing accuracy))
Input - 
  encoder_model
  decoder_model
  encoder_layers
  decoder_layers
Output - Testing accuracy 
'''
def test_accuracy(encoder_model,decoder_model,encoder_layers,decoder_layers):
  #count the number of words that are predicted correctly
  success=0
  #Get all the predicted words
  pred=do_predictions(test_input_tensor,encoder_model,decoder_model,test_input_tensor.shape[0],encoder_layers,decoder_layers)

  for seq_index in range(test_input_tensor.shape[0]):
      predicted_word = pred[seq_index] #predicted_Word
      target_word=test_target_texts[seq_index][1:-1] #target_word_ground_truth
      #test the word one by one and write to files
      #success word
      if target_word == predicted_word:
        success+=1
        f = open("success.txt", "a")
        f.write(test_input_texts[seq_index]+' '+target_word+' '+predicted_word+'\n')
        f.close()
      #failure word (if it is not correct predictions)
      else:
        f = open("failure.txt", "a")
        f.write(test_input_texts[seq_index]+' '+target_word+' '+predicted_word+'\n')
        f.close()
  return float(success)/float(test_input_tensor.shape[0])
  
  
'''
Function - batch_validate (validate entire batch)
Input - 
  encoder_model
  decoder_model
  encoder_layers
  decoder_layers
Output - 
  Return validation accuracy
'''
def batch_validate(encoder_model,decoder_model,encoder_layers,decoder_layers):
  success=0
  #get all the predicted words
  pred=do_predictions(val_input_tensor,encoder_model,decoder_model,val_input_tensor.shape[0],encoder_layers,decoder_layers)
  for seq_index in range(val_input_tensor.shape[0]):
      predicted_word = pred[seq_index] #predicted word
      target_word=val_target_texts[seq_index][1:-1] #groundtruth word (target word)
      #test the words one by one
      if predicted_word == target_word:
        success+=1 #increasing the success 
  return float(success)/float(val_input_tensor.shape[0]) #returning the accuracy
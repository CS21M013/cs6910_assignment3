'''
Imports
'''
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import copy
'''
Function - inferencing
Inputs - 
  model
  encoder_layers
  decoder_layers
Output - encoder model and the deocder model separately
'''
def inferencing(model,encoder_layers,decoder_layers):

    ######################################################################### Encoder Model ###################################################################
    # Defining the encoder_inputs
    e_inputs = model.input[0]  
    # Checking if the model layers are LSTM layers
    if isinstance(model.layers[encoder_layers+3], keras.layers.LSTM):
      e_outputs, state_h_enc, state_c_enc = model.layers[encoder_layers+3].output #geting the encoded output of the layers
      e_states = [state_h_enc, state_c_enc] #getting the both hidden states of the layers
    
    # Checking if the model layers are GRU or RNN layers
    elif isinstance(model.layers[encoder_layers+3], keras.layers.GRU) or isinstance(model.layers[encoder_layers+3], keras.layers.RNN):
      e_outputs, state = model.layers[encoder_layers+3].output #geting the encoded output of the layers 
      e_states = [state] #getting the hidden states of the layers
    
    #Genrating the encoder model
    encoder_model = keras.Model(e_inputs, e_states)

    ########################################################################### Decoder Model ####################################################################
    #defining the decoder inputs
    d_inputs =  keras.Input(shape=( 1))  
    # Checking if the model layers were LSTM layers
    if isinstance(model.layers[encoder_layers+3], keras.layers.LSTM):
      decoder_states_inputs=[]
      d_states=[]
      last=None
      for i in range(decoder_layers):
        #every layer must have an input through which we can supply it's hidden state
        decoder_state_input_h = keras.Input(shape=(latent_dim,),name='inp3_'+str(i)) #decoder state H
        decoder_state_input_c = keras.Input(shape=(latent_dim,),name='inp4_'+str(i)) #decoder state C
        init = [decoder_state_input_h, decoder_state_input_c] #state containing both H and C
        decoder_lstm = model.layers[i+encoder_layers+4]
        #If it is the first decoder layer
        if i==0:
          d_outputs, state_h, state_c = decoder_lstm(
              model.layers[i+encoder_layers+2](d_inputs), initial_state=init
          )
        # Consecutive decoding layers
        else:
          d_outputs, state_h, state_c = decoder_lstm(
              last, initial_state=init 
          )
        #saving the final deocder outputs as last output.
        last=d_outputs
        #appending the input states and the hidden states at every layer
        decoder_states_inputs.append (decoder_state_input_h)
        decoder_states_inputs.append (decoder_state_input_c)
        d_states.append (state_h)
        d_states.append (state_c)

    # Checking if the model layers were GRU or RNN layers
    elif isinstance(model.layers[encoder_layers+3], keras.layers.GRU) or isinstance(model.layers[encoder_layers+3], keras.layers.RNN):
      decoder_states_inputs=[] 
      d_states=[] 
      last=None
      #every layer must have an input through which we can supply it's hidden state
      for i in range(decoder_layers):
        decoder_state_input = keras.Input(shape=(latent_dim,),name='inp3_'+str(i)) #decoder state
        init = [decoder_state_input] #state
        decoder_lstm = model.layers[i+encoder_layers+4]
        #If it is the first decoder layer
        if i==0:
          d_outputs, state = decoder_lstm(
              model.layers[i+encoder_layers+2](d_inputs), initial_state=init
          )
        # Consecutive decoding layers
        else:
          d_outputs, state = decoder_lstm(
              last, initial_state=init 
          )
        #saving the final deocder outputs as last output.
        last=d_outputs
        #appending the input states and the hidden states at every layer
        decoder_states_inputs.append (decoder_state_input)
        d_states.append (state)

    '''
    Geting ther dense final layer from the model objective
    '''
    decoder_dense = model.get_layer('final')
    d_outputs = decoder_dense(last) #outputs of the decoder dense layer
    #Finalizing the decoder model.
    decoder_model = keras.Model(
        [d_inputs] + decoder_states_inputs, [d_outputs] + d_states
    )
    #returning the encoder and the decoder model for inferencing during validation of the model
    return encoder_model,decoder_model
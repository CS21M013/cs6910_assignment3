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
    encoder_inputs = model.input[0]  
    # Checking if the model layers are LSTM layers
    if isinstance(model.layers[encoder_layers+3], keras.layers.LSTM):
      encoder_outputs, state_h_enc, state_c_enc = model.layers[encoder_layers+3].output #geting the encoded output of the layers
      encoder_states = [state_h_enc, state_c_enc] #getting the both hidden states of the layers
    
    # Checking if the model layers are GRU layers
    elif isinstance(model.layers[encoder_layers+3], keras.layers.GRU):
      encoder_outputs, state = model.layers[encoder_layers+3].output #geting the encoded output of the layers 
      encoder_states = [state] #getting the hidden states of the layers
    
    # Checking if the model layers are RNN layers
    elif isinstance(model.layers[encoder_layers+3], keras.layers.RNN): 
      encoder_outputs, state = model.layers[encoder_layers+3].output #geting the encoded output of the layers  
      encoder_states = [state] #getting the hidden states of the layers
    #Genrating the encoder model
    encoder_model = keras.Model(encoder_inputs, encoder_states)

    ########################################################################### Decoder Model ####################################################################
    #defining the decoder inputs
    decoder_inputs =  keras.Input(shape=( 1))  
    # Checking if the model layers were LSTM layers
    if isinstance(model.layers[encoder_layers+3], keras.layers.LSTM):
      decoder_states_inputs=[]
      decoder_states=[]
      last=None
      for i in range(decoder_layers):
        #every layer must have an input through which we can supply it's hidden state
        decoder_state_input_h = keras.Input(shape=(latent_dim,),name='inp3_'+str(i)) #decoder state H
        decoder_state_input_c = keras.Input(shape=(latent_dim,),name='inp4_'+str(i)) #decoder state C
        x = [decoder_state_input_h, decoder_state_input_c] #state containing both H and C
        decoder_lstm = model.layers[i+encoder_layers+4]
        #If it is the first decoder layer
        if i==0:
          decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
              model.layers[i+encoder_layers+2](decoder_inputs), initial_state=x
          )
        # Consecutive decoding layers
        else:
          decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
              last, initial_state=x 
          )
        #saving the final deocder outputs as last output.
        last=decoder_outputs
        #appending the input states and the hidden states at every layer
        decoder_states_inputs.append (decoder_state_input_h)
        decoder_states_inputs.append (decoder_state_input_c)
        decoder_states.append (state_h_dec)
        decoder_states.append (state_c_dec)

    # Checking if the model layers were GRU layers
    elif isinstance(model.layers[encoder_layers+3], keras.layers.GRU):
      decoder_states_inputs=[] 
      decoder_states=[] 
      last=None
      #every layer must have an input through which we can supply it's hidden state
      for i in range(decoder_layers):
        decoder_state_input = keras.Input(shape=(latent_dim,),name='inp3_'+str(i)) #decoder state
        x = [decoder_state_input] #state
        decoder_lstm = model.layers[i+encoder_layers+4]
        #If it is the first decoder layer
        if i==0:
          decoder_outputs, state = decoder_lstm(
              model.layers[i+encoder_layers+2](decoder_inputs), initial_state=x
          )
        # Consecutive decoding layers
        else:
          decoder_outputs, state = decoder_lstm(
              last, initial_state=x 
          )
        #saving the final deocder outputs as last output.
        last=decoder_outputs
        #appending the input states and the hidden states at every layer
        decoder_states_inputs.append (decoder_state_input)
        decoder_states.append (state)

    # Checking if the model layers were RNN layers
    elif isinstance(model.layers[encoder_layers+3], keras.layers.RNN):
      decoder_states_inputs=[]
      decoder_states=[]
      last=None
      #every layer must have an input through which we can supply it's hidden state
      for i in range(decoder_layers):
        decoder_state_input = keras.Input(shape=(latent_dim,),name='inp3_'+str(i)) #decoder state
        x = [decoder_state_input] #state
        decoder_lstm = model.layers[i+encoder_layers+4]
        #If it is the first decoder layer
        if i==0:
          decoder_outputs, state = decoder_lstm(
              model.layers[i+encoder_layers+2](decoder_inputs), initial_state=x
          )
        # Consecutive decoding layers
        else:
          decoder_outputs, state = decoder_lstm(
              last, initial_state=x 
          )
        #saving the final deocder outputs as last output.
        last=decoder_outputs
        #appending the input states and the hidden states at every layer
        decoder_states_inputs.append (decoder_state_input)
        decoder_states.append (state)      

    '''
    Geting ther dense final layer from the model objective
    '''
    decoder_dense = model.get_layer('final')
    decoder_outputs = decoder_dense(last) #outputs of the decoder dense layer
    #Finalizing the decoder model.
    decoder_model = keras.Model(
        [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states
    )
    #returning the encoder and the decoder model for inferencing during validation of the model
    return encoder_model,decoder_model
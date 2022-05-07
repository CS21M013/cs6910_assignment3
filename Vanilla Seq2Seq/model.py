'''
Imports
'''
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import copy
'''
Function - Build Model
Input - 
  The RNN cell type
  embeding dimensions
  no of encoder layers
  no of decoder layers
  dropout
Output - It returns the model object
'''
#Building the model
def build_model(rnn_type,embedding_dim,encoder_layers,decoder_layers,dropout):

  '''
  Building the Encoder
  '''

  #Specifying the dimensions of the input layer and initializing it
  encoder_inputs = keras.Input(shape=( max_encoder_seq_length))
  #initialization of the embeding layer
  embed = keras.layers.Embedding(num_encoder_tokens, embedding_dim)(encoder_inputs)
  
  #Adding multiple layers
  last_encoder=None #save the last encoder output for adding mutiple layers.

  #######################################################################  LSTM Encoder ##################################################################### 
  if rnn_type=='LSTM':
    #adding everything except the last LSTM layer, because in last layer return state=True
    for i in range(encoder_layers-1):
      encoder = keras.layers.LSTM(latent_dim, return_sequences=True,dropout=dropout) #Keras LSTM layer adding
      if i==0:
        encoder_out = encoder(embed)  #encoder the first layer
      else:
        encoder_out = encoder(last_encoder) #encode the last layer output for the next layer.
      last_encoder=encoder_out
    #Adding the last layer
    encoder = keras.layers.LSTM(latent_dim, return_state=True,dropout=dropout)
    ''' For only one encoder '''
    if encoder_layers == 1:
      encoder_outputs, state_h, state_c = encoder(embed)
    else:
      encoder_outputs, state_h, state_c = encoder(last_encoder)
    encoder_states = [state_h, state_c] #storing both the hidden states.

  #######################################################################  GRU or RNN Encoder ##################################################################### 
  elif rnn_type=='GRU' or rnn_type=="RNN":
    #adding everything except the last GRU layer, because in last layer return state=True    
    for i in range(encoder_layers-1):
      if rnn_type=="GRU":
          encoder = keras.layers.GRU(latent_dim, return_sequences=True,dropout=dropout) #keras GRU layer
          if i==0:
            encoder_out = encoder(embed) #encode the first layer
          else:
            encoder_out = encoder(last_encoder) #encode the last layer output for the next layer
      elif rnn_type=="RNN":
          encoder = keras.layers.SimpleRNN(latent_dim, return_sequences=True,dropout=dropout)
          if i==0:
            encoder_out = encoder(embed) #Encode the first layer
          else:
            encoder_out = encoder(last_encoder) #encode the last layer output for the next layer
      last_encoder=encoder_out
    #Adding the last layer
    encoder = keras.layers.GRU(latent_dim, return_state=True,dropout=dropout)
    '''If there is only one encoder'''
    if encoder_layers == 1:
      encoder_outputs, state = encoder(embed)
    else:
      encoder_outputs, state = encoder(last_encoder)
    encoder_states = [state] #Storing the encoder hidden state

  '''
  Building the Deocder
  '''
  #specifying the dimension of the input layer and initializing it
  decoder_inputs = keras.Input(shape=( max_decoder_seq_length))
  #initializing the embedding layer
  embed = keras.layers.Embedding(num_decoder_tokens, embedding_dim)(decoder_inputs)

  ######################################################################## LSTM Decoder #########################################################################
  if rnn_type=="LSTM":
    #adding all the LSTM layers
    for i in range(decoder_layers):
      decoder = keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True,dropout=dropout) #Keras LSTM layer
      if i==0:
        decoder_outputs, _, _ = decoder(embed, initial_state=encoder_states) #getting the decoder output for the first decoder using embed
      else:  
        decoder_outputs, _, _ = decoder(last, initial_state=encoder_states) #getting the decoder output for the remaining decoders
      #geting the output from the last decoder
      last=decoder_outputs

    #Adding dense layer at the end
    decoder_dense = keras.layers.Dense(num_decoder_tokens, activation="softmax",name='final') #softmax dense function.
    decoder_outputs = decoder_dense(last) #geting the final decoder outputs by calling the dense layer.

  ######################################################################## GRU or RNN Decoder #########################################################################
  elif rnn_type=="GRU" or rnn_type=="RNN":
    #adding all the GRU layers
    for i in range(decoder_layers):
      if rnn_type=="GRU":
          decoder = keras.layers.GRU(latent_dim, return_sequences=True, return_state=True,dropout=dropout) #Keras GRU layer
          if i==0:
            decoder_outputs, _= decoder(embed, initial_state=encoder_states) #getting the decoder output for the first decoder using embed
          else:  
            decoder_outputs, _ = decoder(last, initial_state=encoder_states) #getting the decoder output for the remaining decoders
      elif rnn_type=="RNN":
          decoder = keras.layers.SimpleRNN(latent_dim, return_sequences=True, return_state=True,dropout=dropout) #Keras RNN layer
          if i==0:
            decoder_outputs, _= decoder(embed, initial_state=encoder_states) #getting the decoder output for the first decoder using embed
          else:  
            decoder_outputs, _ = decoder(last, initial_state=encoder_states) #getting the decoder output for the remaining decoders      
      #geting the output from the last decoder
      last=decoder_outputs

    #Adding dense layer at the end
    decoder_dense = keras.layers.Dense(num_decoder_tokens, activation="softmax",name='final') #softmax dense function.
    decoder_outputs = decoder_dense(last) #geting the final decoder outputs by calling the dense layer.

  #creating the model using the encoder inputs, decoder inputs and the decoder outputs
  model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

  #return the keras Model Object using the defined parameters.
  return model
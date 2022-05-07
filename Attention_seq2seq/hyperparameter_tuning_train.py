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
Downloading the dataset
'''
# Download the dataset
!curl https://storage.googleapis.com/gresearch/dakshina/dakshina_dataset_v1.0.tar --output daksh.tar
# Extract the downloaded tar file
!tar -xvf  'daksh.tar'
# Set the file paths to train, validation and test dataset
#train_path
train_file_path=os.path.join(os.getcwd(),"dakshina_dataset_v1.0","hi","lexicons","hi.translit.sampled.train.tsv")
#validation_path
vaildation_file_path = os.path.join(os.getcwd(),"dakshina_dataset_v1.0","hi","lexicons","hi.translit.sampled.dev.tsv")
#test_path
test_file_path = os.path.join(os.getcwd(),"dakshina_dataset_v1.0","hi","lexicons","hi.translit.sampled.test.tsv")

'''
Reading the training dataset entirely
'''
# Use the entire training dataset file
input_tensor_train, target_tensor_train, inp_lang, targ_lang = load_dataset(train_file_path)

# Calculate max_length of the target tensors
max_length_targ, max_length_inp = target_tensor_train.shape[1], input_tensor_train.shape[1]

def train(use_wandb=True):
    global BATCH_SIZE 
    global units 
    global vocab_inp_size
    global vocab_tar_size
    global embedding_dim
    global encoder
    global decoder
    global optimizer
    global loss_object
    global checkpoint_dir
    global checkpoint_prefix 
    global checkpoint
    global run_name
    global rnn_type


    '''
    Wandb configuration
    '''
    # initialising the wandb run
    run = wandb.init()
    # Tpye of RNN to choose. Acceptable Values are 'RNN'. 'LSTM' and 'GRU'
    rnn_type = run.config.rnn_type
    # Batch size for training.
    BATCH_SIZE = run.config.bs
    # Dimensions of the abstract representation of the input word and target word.
    embedding_dim = run.config.embed
    # Latent dimensions of the encoder and decoder.
    units = run.config.latent
    # Number of epochs to train for.
    EPOCHS = run.config.epochs
    #	Float between 0 and 1. Denotes the fraction of the units to drop.
    dropout = run.config.dropout


    print("rnn_Type: ",rnn_type)
    #buffer size
    BUFFER_SIZE = len(input_tensor_train)
    #steps per epoch
    steps_per_epoch = len(input_tensor_train)//BATCH_SIZE
    #vocab input size
    vocab_inp_size = len(inp_lang.word_index)+1
    #vocab target size
    vocab_tar_size = len(targ_lang.word_index)+1
    
    #wandb run name
    run_name = '_epochs_'+str(EPOCHS)+'_rnn_type_'+str(rnn_type)+'_bs_'+str(BATCH_SIZE)+'_embed_'+str(embedding_dim)+'_latent_'+str(units)+'_dropout_'+str(dropout)
    if use_wandb==True:
      wandb.run.name = run_name

    dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
    # We create batches of size BATCH_SIZE and ignore the last batch because the last batch may not be equal to BATCH_SIZE
    #creating the batches
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    
    train_input_batch, train_target_batch = next(iter(dataset))
    
    encoder, decoder = build_model(vocab_inp_size, vocab_tar_size, embedding_dim, units, BATCH_SIZE, dropout,train_input_batch)
  
    #Apply the adam optimizer
    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    
    #saving the checkpoints
    checkpoint_dir = os.path.join(os.getcwd(),'training_checkpoints')
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)
        
    ####################################################################### Hyper parameter Tuning (Training) ###############################################################
    
    train_loss=[0]*EPOCHS
    for epoch in range(EPOCHS):
      start = time.time()
      if rnn_type!='LSTM': #GRU or RNN
        enc_hidden = encoder.initialize_hidden_state()
      elif rnn_type=='LSTM': #LSTM
        enc_hidden,enc_cell_state = encoder.initialize_hidden_state()
      total_loss = 0
      for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
        #Training for every batch
        if rnn_type!='LSTM':
          batch_loss = train_batch(inp, targ, enc_hidden, encoder,decoder,rnn_type)
        elif rnn_type=='LSTM':
          batch_loss = train_batch(inp, targ, [enc_hidden,enc_cell_state], encoder,decoder,rnn_type)
        total_loss += batch_loss
      if batch % 100 == 0:
        #printing the batch loss
        print(f'Epoch {epoch+1} Batch {batch} Loss {batch_loss.numpy():.4f}')
        
      # saving (checkpoint) the model every 2 epochs
      if (epoch + 1) % 2 == 0:
        checkpoint.save(file_prefix=checkpoint_prefix)
      #print the trtaining loss for the epoch
      print(f'Epoch {epoch+1} Loss {total_loss/steps_per_epoch:.4f}')
      print(f'Time taken for 1 epoch {time.time()-start:.2f} sec\n')
      # Storing the average loss per epoch
      train_loss[epoch] = total_loss.numpy()/steps_per_epoch
      #logging the training loss in wandb
      if use_wandb == True:
        wandb.log({"train_loss": total_loss.numpy()/steps_per_epoch})

        
    #test_accuracy = validate(test_file_path,run_name)
    val_acc=validate(vaildation_file_path,rnn_type)
    print("Train loss: ",train_loss)
    print("Validation Accuracy: ",val_acc)
    #print("Test Accuracy: ",test_accuracy)
    if use_wandb ==True:
      wandb.log({'val_accuracy': val_acc})
    
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

'''
Sweep configuration for hyper parameter tuning
'''
!pip install wandb --upgrade
import wandb
!wandb login
sweep_config = {
    "name": "Bayesian Sweep without attention",
    "method": "bayes",
    "metric": {"name": "val_accuracy", "goal": "maximize"},
    "parameters": {
        
        "rnn_type": {"values": ["RNN","GRU","LSTM"]},
        
        "embed": {"values": [256,512]},
        
        "latent": {"values": [512,1024]},
        
        "dropout": {"values": [0.1, 0.2, 0.3]},

        "epochs": {"values": [20]},
        
        "bs": {"values": [64]},


    },
  }

sweep_id = wandb.sweep(sweep_config, project="CS6910_Assignment3_attention", entity="cs21m007_cs21m013")

wandb.agent(sweep_id, train, count = 30)
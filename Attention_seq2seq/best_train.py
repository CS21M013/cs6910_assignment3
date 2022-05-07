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

'''
Function for training manually the best configuration of the model
'''
def manual_train():
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
    Best configuration of the model
    '''
    rnn_type = 'LSTM'
    BATCH_SIZE = 64
    embedding_dim = 512
    units = 1024
    EPOCHS = 20
    dropout = 0.2

    print("rnn_Type: ",rnn_type)
    #generating the buffer size
    BUFFER_SIZE = len(input_tensor_train)
    #calculating the number of steps per epoch
    steps_per_epoch = len(input_tensor_train)//BATCH_SIZE
    #vocab input size
    vocab_inp_size = len(inp_lang.word_index)+1
    #vocab target size
    vocab_tar_size = len(targ_lang.word_index)+1

    run_name = '_epochs_'+str(EPOCHS)+'_rnn_type_'+str(rnn_type)+'_bs_'+str(BATCH_SIZE)+'_embed_'+str(embedding_dim)+'_latent_'+str(units)+'_dropout_'+str(dropout)
    
    dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
    #We are creating batches of size BATCH_SIZE and ignore the last batch because the last batch may not be equal to BATCH_SIZE
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    
    train_input_batch, train_target_batch = next(iter(dataset))
    
    encoder, decoder = build_model(vocab_inp_size, vocab_tar_size, embedding_dim, units, BATCH_SIZE, dropout,train_input_batch)

    #apply the adam optimizer
    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    
    #creating and saving the checkpoints
    checkpoint_dir = os.path.join(os.getcwd(),'training_checkpoints')
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)
    
    train_loss=[0]*EPOCHS
    
    ############################################################################## Training ##########################################################################

    train_loss = train_epochs(EPOCHS,encoder,decoder,dataset,steps_per_epoch)

        
    #calculating the test accuracy using the validate function
    test_accuracy = validate(test_file_path,run_name)
    #calcualting the validation accuracy using validate function
    val_acc=validate(vaildation_file_path,rnn_type)
    print("Train loss: ",train_loss)
    print("Validation Accuracy: ",val_acc)
    print("Test Accuracy: ",test_accuracy)
    
 	  # restoring the latest checkpoint in checkpoint_dir and starting the test
  	# checkpoints are only useful when source code that will use the saved parameter values is available.
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    generate_inputs(rnn_type,10)
    #generating the connectivity of the best model.
    connectivity(['maryaadaa','prayogshala','angarakshak'],rnn_type, os.path.join(os.getcwd(),"predictions_attention",str(run_name)))
	
'''
manual training for the best parameter model
'''
manual_train()

#Download a copy of the predictions_attention and training_checkpoints folder.
!zip -r /content/predictions_attention.zip /content/predictions_attention
!zip -r /content/training_checkpoints.zip /content/training_checkpoints
from google.colab import files
files.download("/content/predictions_attention.zip")
files.download("/content/training_checkpoints.zip")
	

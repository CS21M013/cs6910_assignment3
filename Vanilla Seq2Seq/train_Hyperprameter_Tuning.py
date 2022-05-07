'''
Imports
'''
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import copy
from model import build_model
from inference import inferencing
from predict import do_predictions
from util import test_accuracy, batch_validate
'''
Downloading the data
'''
!curl https://storage.googleapis.com/gresearch/dakshina/dakshina_dataset_v1.0.tar --output daksh.tar

'''
Capturing the data and saving as the Tar file
'''
!tar -xvf  'daksh.tar' 

'''
Function to read the data
Input - Data path to read the data
Output - input text, target text, input and target tokenizier, input and target tensor
'''
def data(path,input_tokenizer=None,output_tokenizer=None,input_length=None,output_length=None):
  
  input_texts = []  #list of input text
  output_texts = [] #list of output/target text
  
  df = pd.read_csv(path,sep="\t",names=["1", "2","3"]).astype(str)
  # sampling the input of the tokenizier in None.
  if input_tokenizer is None:
      df=df.sample(frac=1)
  # Adding all the  input and target texts with start sequence and end sequence added to target. 
  for index, row in df.iterrows():
      input_text=row['2']
      output_text= row['1']
      if output_text =='</s>' or input_text=='</s>': #adding the start character for input and output text
        continue
      output_text = "\t" + output_text + "\n" #addintg the ending character for both input an the output
      input_texts.append(input_text)
      output_texts.append(output_text)
  
  #only train set will have input_tokenizer as none. Validation and test will will use the same.
  if input_tokenizer is None:
    input_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', char_level=True)
    input_tokenizer.fit_on_texts(input_texts)
  input_tensor = input_tokenizer.texts_to_sequences(input_texts) #generating the input tensor
  #performing pading on the input sequences
  input_tensor = tf.keras.preprocessing.sequence.pad_sequences(input_tensor,padding='post')
  
  if output_tokenizer is None:
    output_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', char_level=True)
    output_tokenizer.fit_on_texts(output_texts)
  #generating the target tensor
  output_tensor = output_tokenizer.texts_to_sequences(output_texts)
  output_tensor = tf.keras.preprocessing.sequence.pad_sequences(output_tensor,padding='post')
  #for dataset which is not training (validation and the testing) we pad to make maximum length same as train set.
  if input_length is not None and output_length is not None:
      input_tensor=tf.concat([input_tensor,tf.zeros((input_tensor.shape[0],input_length-input_tensor.shape[1]))],axis=1)
      output_tensor=tf.concat([output_tensor,tf.zeros((output_tensor.shape[0],output_length-output_tensor.shape[1]))],axis=1)
  #returning the input and output tokenizer, text and the tensors.
  return input_texts,input_tensor,input_tokenizer,output_texts,output_tensor,output_tokenizer
  
# Preprocessing and reading the training data
%%capture
input_texts,input_tensor,input_tokenizer,target_texts,target_tensor,target_tokenizer=data("/content/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.train.tsv")

# Preprocessing and reading the validation data
%%capture
val_input_texts,val_input_tensor,val_input_tokenizer,val_target_texts,val_target_tensor,val_target_tokenizer=data("/content/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.dev.tsv",input_tokenizer,target_tokenizer,input_tensor.shape[1],target_tensor.shape[1])

# Preprocessing and reading the testing data
%%capture
test_input_texts,test_input_tensor,test_input_tokenizer,test_target_texts,test_target_tensor,test_target_tokenizer=data("/content/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.test.tsv",input_tokenizer,target_tokenizer,input_tensor.shape[1],target_tensor.shape[1])

num_encoder_tokens = len(input_tokenizer.word_index)+1  #number of encoder tokens
num_decoder_tokens = len(target_tokenizer.word_index)+1 #number of deccoder tokens
max_encoder_seq_length =  input_tensor.shape[1]         #encoder sequence length
max_decoder_seq_length = target_tensor.shape[1]         #deocoder sequence length

#converting the index to character
index_to_char_input = dict((input_tokenizer.word_index[key], key) for key in input_tokenizer.word_index.keys())     #index to input character
index_to_char_target = dict((target_tokenizer.word_index[key], key) for key in target_tokenizer.word_index.keys())  #index to output/target character

#defining globals
rnn_type=None
embedding_dim=None
model= None
latent_dim = None
enc_layers=None
dec_layers=None
'''
Function- train()
Performs the entire training using Wandb sweeps
'''
def train():
  global rnn_type
  global embedding_dim
  global model
  global latent_dim
  global enc_layer
  global dec_layer
  #intializing wandb
  wandb.init()
  #initializing the wandb configurations
  rnn_type=wandb.config.rnn_type              #RNN cell type
  embedding_dim=wandb.config.embedding_dim    #embedding dimensions
  latent_dim=wandb.config.latent_dim          #latent dimensions
  enc_layer=wandb.config.enc_layer            #encoder_layer
  dec_layer=wandb.config.dec_layer            #decoder layers
  dropout=wandb.config.dropout                #dropout
  epochs=wandb.config.epochs                  #epochs
  bs=wandb.config.bs                          #batch size

  #wandb run name initialization
  wandb.run.name = 'epochs_'+str(epochs)+'_bs_'+str(bs)+'_rnn_type_'+str(rnn_type)+'_em_'+str(embedding_dim)+'_latd_'+str(latent_dim)+'_encs_'+str(enc_layer)+'_decs_'+str(dec_layer)+'_dr_'+str(dropout)

  #building the model
  model=build_model(rnn_type=rnn_type,embedding_dim=embedding_dim,encoder_layers=enc_layer,decoder_layers=dec_layer,dropout=dropout)

  #model compilation
  model.compile(
      optimizer="adam", loss=keras.losses.SparseCategoricalCrossentropy(
                                                              reduction='none'), metrics=["accuracy"]
  )
  ############################################################################### Training the model ######################################################################
  for i in range(epochs):
    hist=model.fit(
        [input_tensor, target_tensor],
        tf.concat([target_tensor[:,1:],tf.zeros((target_tensor[:,:].shape[0],1))], axis=1),
        batch_size=bs,
        epochs=1,shuffle=True
    )
    # Save model
    model.save("s2s.keras")
    ############################################################################## Inferencing ##############################################################################
    # Restore the model and construct the encoder and decoder.
    inf = keras.models.load_model("/content/s2s.keras")
    encoder_model,decoder_model=inferencing(inf,encoder_layers=enc_layer,decoder_layers=dec_layer)
    #log train loss to wandb
    wandb.log({"train_loss": hist.history['loss'][0]})
  #calculate the validation accuracy
  val_acc=batch_validate(encoder_model,decoder_model,enc_layer,dec_layer)
  #logging the validation accuracy
  wandb.log({"val_acc":val_acc})
  
'''
Wandb details importing and log in
'''
%%capture
!pip install wandb
wb=True
import wandb
if wb:
  wandb.login()
  
# generating the wandb sweep configuration
if wb:
  sweep_config = {
    "name": "Bayesian Sweep without attention",
    "method": "bayes", #method used was bayesian
    "metric": {"name": "val_acc", "goal": "maximize"}, #mximizing the validation accuracy
    "parameters": {
        
        "rnn_type": {"values": ["GRU","LSTM"]},
        
        "embedding_dim": {"values": [128,256,512]},
        
        "latent_dim": {"values": [128,256,512,1024]},
        
        "enc_layer": {"values": [1, 2, 3]},
        
        "dec_layer": {"values": [1, 2, 3]},
        
        "dropout": {"values": [0.1, 0.2, 0.3]},

        "epochs": {"values": [20]},
        
        "bs": {"values": [64]},


    },
  }
  #creating the wandb sweep
  sweep_id = wandb.sweep(sweep_config, project="CS6910_Assignment3_vanilla", entity="cs21m007_cs21m013")
  #calling the wandb sweep to start the hyper parameter tuning.
  wandb.agent(sweep_id, train, count = 30)
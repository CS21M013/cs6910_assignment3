# Attention based Seq2Seq Model
## The attention based transliteration model developed using RNNs (RNN,LSTM and GRU)
The repository contains the following files and folders.
1. cs6910_assignment3_attention.ipynb - The .ipynb file contains the entire executable functions for developing the attention based transliteration model and evaluating and visualizing the attention weights and the connectivity.
2. attention.py - This python file implements the attention mechanism referenced at https://arxiv.org/abs/1409.0473
3. dataset_util.py - This python file contains function that aids in reading the dataset and the loading the dataset
4. decoder.py - This python file contains the class of the decoders (RNN,LSTM and GRU).
5. encoder.py - This python file contains the class of the encoders (RNN, LSTM and GRU).
6. train_util.py - This python file contains functions that aid in the training of the model.
7. prediction_plots_util.py - This python file contains the functions that help in ploting the attention weights and the connectivity.
8. hyperparameter_tuning_train.py - This python file contains the function to perform the hyper-paramater tuning or sweeping to get the best parameters of the model.
9. best_train.py - This python file contains the manual_train function to train and evaluate the best model after hyper-paramter tuning.

### Requirements for running the code
```
Python 
Tensorflow
pandas
numpy
matplotlib
```
### Execution of the code
1. Run the below mentioned .ipynb code entirely to perform the model training with the best set of hyper-paramaters. The cells corresponding to the wandb sweeping need not be run.
    ```
    cs6910_assignment3_attention.ipynb
    ```
2. Download the Github Repository -> Keep all the files in the same folder -> Run the below mentioned .py file as follows:
    ```
    Python3 best_train.py
    ```
    The execution of the best will lead to its evaluation and thus the generation of the following files.  
      1.Attention weights visualization of 10 different words (heatmap visualization)  
      2.Connectivity.html (html file denoting the connectivity among different characters in a word)  
      3.success.txt (correct predictions)  
      4.failure.txt (wrong predictions)

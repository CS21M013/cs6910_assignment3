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
1. Run the below mentioned .ipynb code entirely to perform the model training with the best set of hyper-paramaters. The cells corresponding to the wandb sweeping need not be run. You need to upload the Nirmala.ttf to the google colab before executing the code.
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

### Validation accuracy and the test accuracy of the best trained model
```
Validation Accuracy:  0.41358733073215515
Test Accuracy:  0.40679848922461675
```
### some of the predictions of the model including successful and the failed predictions are as follows - 
Successful predictions
```
Input   True    Predicted
erica एरिका एरिका
erika एरिका एरिका
eliminator एलिमिनेटर एलिमिनेटर
sfc एसएफसी एसएफसी
sk एसके एसके
spf एसपीएफ एसपीएफ
aster एस्टर एस्टर
ester एस्टर एस्टर
asthetic एस्थेटिक एस्थेटिक
ehsas एहसास एहसास
ainth ऐंठ ऐंठ
enth ऐंठ ऐंठ
```
Failed Predictions of the attention based model
```
asaamnjsy असामंजस्य असामजन्य
asamanjasy असामंजस्य असमंजस्य
asamanjasya असामंजस्य असमंजस्य
asset असेट एसेट
asvabhavik अस्वभाविक अस्वाभाविक
aswabhavik अस्वभाविक अस्वाभाविक
aswbhavik अस्वभाविक अस्बाभाविक
item आइटम आइटेम
items आइटम्स आइटेम्स
idle आइडल आईडल
idol आइडल आईडल
icc आइसीसी आईसीसी
iso आइसो आईएसओ
```
### Visualizations of the attention weights
![Screenshot](https://github.com/CS21M013/CS6910_assignment3/blob/main/Attention_seq2seq/visualizations/Untitled%20Diagram.drawio.png)

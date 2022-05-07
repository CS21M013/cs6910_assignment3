# Vanilla Seq2Seq Model
## This is the vanilla transliteration model developed using RNNs (RNN, LSTM and GRU)
The folder contains the follwing files:  
1. CS6910_vanilla_seq2seq.ipynb - The ipynb file containing the entire code for the experimentation. It can be executed entirely on its own.  
2. best_train.py - The python file that is used for the training and evaluation of the best model.
3. inference.py - The python file used for inferencing the model
4. model.py - The python file containg the entire model template using encoder and decoder architecture
5. predict.py - The python file that will be used to get the predictions of the best model on the test data
6. train_Hyperprameter_Tuning.py - The python file is for performing the sweeps or the hyperparameter tuning of the model for some parameters
7. util.py This python file contains some functions that can be used as utilitary functions for the training and the model development and testing of the model.    
8. Predictions folder - Containing the success and the failure text files (predictions of the best model).
## The Best model architecture can be visualized as follows  
![Screenshot](model.png)
## Requirements of running the code
    1. Python
    2. Tensorflow
    3. matplotlib
    4. pandas
    5. numpy
## Execution of the code
1. upload the below .ipynb file and Run every cell except the cells marked as to be run only for HyperParamter Tuning to train and evaluate the best performing model.
    ```
    CS6910_vanilla_seq2seq.ipynb
    ```
2. Download the Github Repository -> Keep all the files in the same folder -> Run the below mentioned .py file as follows.
    ```
    python3 best_train.py
    ```
    The execution of the above files generates the model architecture and the succees and the failure text files and prints the validation accuracy of every epoch and the 
    test accuracy achieved on the test data.

### The validation accuracy and the text accuracy thus can be visualized as folllows;
```
691/691 [==============================] - 164s 237ms/step - loss: 0.0176 - accuracy: 0.9946
Validation Accuracy 0.39031665901789814
Test Accuracy 0.38605064415815193
```
### Some of the sample inputs and the predictions are as folllows:
Thus the model's successful predictions are as follows:
```
Input   True    Predicted
ashriton आश्रितों आश्रितों
aastik आस्तिक आस्तिक
astik आस्तिक आस्तिक
aahnaan आह्नान आह्नान
aahnan आह्नान आह्नान
equities इक्विटीज इक्विटीज
ilaichi इलाइची इलाइची
ilaj इलाज इलाज
ishaaq इशाक इशाक
ishaara इशारा इशारा
ishaaraa इशारा इशारा
```
Now some of the failed predictions are:
```
Input   True    Predicted
asvabhavik अस्वभाविक अस्वाभाविक
aswabhavik अस्वभाविक अस्वाभाविक
aswbhavik अस्वभाविक अस्वाभिक
item आइटम आईटम
items आइटम्स आईटम्स
idle आइडल आईडल
idol आइडल आईडल
icc आइसीसी आईसीसी
iso आइसो आईएसओ
ipad आईपैड इपाद
iris आईरिस आइरिस
ounce आउंस ऊंसे
```


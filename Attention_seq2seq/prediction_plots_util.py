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
These codes have been refered from internet. and the blogs provided in the problem itself.
'''
'''
Function  - plot - attention
Function ploting the attention plots.
'''
def plot_attention(attention, input_word, predicted_word, file_name):
  #loading the hindi font for displaying
  hindi_font = FontProperties(fname = os.path.join(os.getcwd(),"Nirmala.ttf"))
  
  #figure matplotlib
  fig = plt.figure(figsize=(3, 3))
  ax = fig.add_subplot(1, 1, 1)
  ax.matshow(attention, cmap='viridis')
  
  fontdict = {'fontsize': 14}

  ax.set_xticklabels([''] + list(input_word), fontdict=fontdict, rotation=0)
  ax.set_yticklabels([''] + list(predicted_word), fontdict=fontdict,fontproperties=hindi_font)

  ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
  ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

  #save the plot figure.
  plt.savefig(file_name)
  plt.show()
  
'''
Geting the connectivity html file.
'''
# get html element
def cstr(s, color='black'):
	if s == ' ':
		return "<text style=color:#000;padding-left:10px;background-color:{}> </text>".format(color, s)
	else:
		return "<text style=color:#000;background-color:{}>{} </text>".format(color, s)
	
# print html
def print_color(t):
	display(html_print(''.join([cstr(ti, color=ci) for ti,ci in t])))

# get appropriate color for value
# Darker shades of green denotes higher importance.
def get_clr(value):
	colors = ['#85c2e1', '#89c4e2', '#95cae5', '#99cce6', '#a1d0e8',
		'#b2d9ec', '#baddee', '#c2e1f0', '#eff7fb', '#f9e8e8',
		'#f9e8e8', '#f9d4d4', '#f9bdbd', '#f8a8a8', '#f68f8f',
		'#f47676', '#f45f5f', '#f34343', '#f33b3b', '#f42e2e']
	value = int((value * 100) / 5)
	return colors[value]


'''
Function - Visualize the connectivity plots in the HTMl file
'''
def visualize(input_word, output_word, att_w):
  for i in range(len(output_word)):
    print("\nOutput character:", output_word[i], "\n")
    text_colours = []
    for j in range(len(att_w[i])):
      text = (input_word[j], get_clr(att_w[i][j]))
      text_colours.append(text)
    print_color(text_colours)
	
'''
Code for connectivity visualisation.
'''
# get appropriate color for value
# Darker shades of green denotes higher importance.
def get_shade_color(value):
	colors = ['#00fa00', '#00f500',  '#00eb00', '#00e000',  '#00db00',  
           '#00d100',  '#00c700',  '#00c200', '#00b800',  '#00ad00',  
           '#00a800',  '#009e00',  '#009400', '#008f00',  '#008500',
           '#007500',  '#007000',  '#006600', '#006100',  '#005c00',  
           '#005200',  '#004d00',  '#004700', '#003d00',  '#003800',  
           '#003300',  '#002900',  '#002400',  '#001f00',  '#001400']
	value = int((value * 100) / 5)
	return colors[value]

#creating the HTMl file
def create_file(text_colors,input_word,output_word,file_path=os.getcwd()):
  text = '''
  <!DOCTYPE html>
  <html>
  <head>
    <meta charset="UTF-8"> 
    <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js"></script>
    <script>
            $(document).ready(function(){
            var col =['''
  for k in range(3):
      for i in range(len(output_word)):
              text=text+'''['''
              for j in range(len(text_colors[k][i])-1):
                text=text+'''\"'''+text_colors[k][i][j]+'''\"'''+''','''
              text=text+'''\"'''+text_colors[k][i][len(text_colors[k][i])-1]+'''\"'''+'''],'''
  text=text[0:-1]
  text=text+'''];\n'''
  
  for k in range(3):
      for i in range(len(output_word[k])):
            text=text+'''$(\".h'''+str(k)+str(i)+'''\").mouseover(function(){\n'''
            for j in range(len(input_word[k])):
                       text=text+'''$(\".t'''+str(k)+str(j)+'''\").css(\"background-color\", col['''+str(i)+''']'''+'''['''+str(j)+''']);\n'''
            text=text+'''});\n'''
            text=text+'''$(\".h'''+str(k)+str(i)+'''\").mouseout(function(){\n'''
            for l in range(3):
              for j in range(len(input_word[l])):
                text=text+'''$(\".t'''+str(l)+str(j)+'''\").css(\"background-color\", \"#ffff99\");\n'''
            text=text+'''});\n'''
  text=text+'''});\n
</script>
  </head>
      <body>
          <h1>Connectivity:</h1>
          <p> The connection strength between the target for the selected character and the input characters is highlighted in green (reset). Hover over the text to change the selected character.</p>
          <div style="background-color:#ffff99;color:black;padding:2%; margin:4%;">
          <p>
          <div> Output: </div>
          <div style='display:flex; border: 2px solid #d0cccc; padding: 8px; margin: 8px;'>
          '''
  for k in range(3):
      for i in range(len(output_word[k])):
            text=text+'''\n'''+'''\t'''+'''<div class="h'''+str(k)+str(i)+'''\">'''+output_word[k][i]+'''</div>'''
      text=text+'''</div>'''+'\n'+'\t'+'''<div>  </p>'''+'\n'+'\t'+'''<p>
      <div> Input: </div>
      <div style='display:flex; border: 2px solid #d0cccc; padding: 8px; margin: 8px;'>'''    
      for j in range(len(input_word[k])):
        text=text+'''\n'''+'''\t'''+'''<div class="t'''+str(k)+str(j)+'''\">'''+input_word[k][j]+'''</div>'''
      if k<2:
          text = text+'''</div></p></div><p></p></div>
          <div style="background-color:#ffff99;color:black;padding:2%; margin:4%;">
          <div> Output: </div>
          <div style='display:flex; border: 2px solid #d0cccc; padding: 8px; margin: 8px;'>'''
  text=text+'''
        </div>
        </p>
        </div>
        </body>
  </html>
  '''
  fname = os.path.join(file_path,"connectivity.html")
  file = open(fname,"w")
  file.write(text)
  file.close()

#main file to generate the connectivity of HTML file
def connectivity(input_words,rnn_type,file_path):
  #color list
  color_list=[]
  #input word list
  input_word_list=[]
  #output word list
  output_word_list=[]

  for k in range(3):
    #do inferencing in the model
    output_word, input_word, _ ,att_w = inference_model(input_words[k],rnn_type)
    text_colours=[]
    for i in range(len(output_word)):
      colour=[]
      for j in range(len(att_w[i])):
        value=get_shade_color(att_w[i][j])
        colour.append(value)
      text_colours.append(colour)
    #creating the color list
    color_list.append(text_colours)
    #input word list
    input_word_list.append(input_word)
    #output word list
    output_word_list.append(output_word)
  #create file for generating the HTML file
  create_file(color_list,input_word_list,output_word_list,file_path)
  
'''
Function - Transliteration
generating the attention heatmap
'''
def transliterate(input_word,rnn_type,file_name=os.path.join(os.getcwd(),"attention_heatmap.png"),visual_flag=True):
  #do inferencing ot get the predicted word and other attention plots, weights and input word
  predicted_word, input_word, attention_plot,att_w = inference_model(input_word,rnn_type)

  #geting the predicted transliterations
  print("\n",'Input:', input_word)
  print('Predicted transliteration:', predicted_word)

  attention_plot = attention_plot[:len(predicted_word),
                                  :len(input_word)]
  plot_attention(attention_plot, input_word, predicted_word, file_name)

  if visual_flag == True:
    #visualize the attention plots
    visualize(input_word, predicted_word, att_w)
	
'''
Fucntion - generate_inputs
generate the predictions for the attention based model
'''
def generate_inputs(rnn_type,n_test_samples=10):
  target_words, input_words = create_dataset(test_file_path)
  
  for i in range (n_test_samples):
    index = random.randint(0,len(input_words))
    input_word=input_words[index]
    file_name=os.path.join(os.getcwd(),"predictions_attention",str(run_name),input_word+".png")
    
    if i == 0:
      transliterate(input_word[1:-1],rnn_type, file_name,True)
    elif i > 0:
      transliterate(input_word[1:-1],rnn_type, file_name,False)
import os
import re
import codecs
import numpy as np
import pandas as pd

os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']=''
from keras.preprocessing.text import Tokenizer

FILE_PATH = '../input'

# load data
def read_comments(file_name):
  file_name = f'{FILE_PATH}/plain_text/{file_name}.txt'
  text_list = [x for x in open(file_name, encoding='utf-8')]  
  pattern = re.compile(u' |\n')
  text = {word for text in text_list for word in re.split(pattern, text.strip())}
  return text

# tokenize words
def tokenize(text_cleaned):
  word_tokenizer = Tokenizer(filters='', lower=False)
  word_tokenizer.fit_on_texts(text_cleaned)
  return list(word_tokenizer.word_index.keys())

# save data
def save_tokens(tokens):
  save_name = f'{FILE_PATH}/tokens/toxic_tokens.txt'
  file = codecs.open(save_name, 'w', 'utf-8-sig')
  for i in tokens: file.write(i +'\n')
  file.close()  

# iterate through cleaned data
print(f'tokenizing data')
text = read_comments('toxic_comments')
tokens = tokenize(text)
save_tokens(tokens)

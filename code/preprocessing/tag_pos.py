import re
import nltk
import regex
import pandas as pd
import numpy as np
import multiprocessing as mp
from toxic.text_utils import read_comments

# load data
FILE_DIR = '../input'
text = read_comments(f'{FILE_DIR}/plain_text/toxic_comments.txt')
text = pd.Series(text)

# fill in missing
text[text.map(len)<=1] = 'neutral'

# map to pos
def get_pos(x):
  tokens = nltk.tokenize.word_tokenize(x)
  tags = nltk.pos_tag(tokens)
  _, pos = zip(*tags) 
  return ' '.join(pos)

pool = mp.Pool(20)
pos = pool.map(get_pos, text)
pool.terminate()  

# save results
part_of_speech = pd.DataFrame({'pos': pos})
part_of_speech.to_csv(f'{FILE_DIR}/part_of_speech/toxic_pos.csv', index=False)

train = pd.DataFrame({'pos': pos[:159571]})
train.to_csv(f'{FILE_DIR}/part_of_speech/train_pos.csv', index=False)

test = pd.DataFrame({'pos': pos[159571:]})
test.to_csv(f'{FILE_DIR}/part_of_speech/test_pos.csv', index=False)
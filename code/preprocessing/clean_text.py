import re
import pandas as pd

# text cleaning
def clean_text(text):
    text = re.sub(r"\W+", " ", text.lower())
    text = bytes(text, encoding="utf-8")
    text = text.replace(b"\n", b" ")
    text = text.replace(b"\t", b" ")
    text = text.replace(b"\b", b" ")
    text = text.replace(b"\r", b" ")
    text = regex.sub(b"\s+", b" ", text)
    return str(text, 'utf-8')

FILE_DIR = '../input'
    
# load and process data
train_text = pd.read_csv(f'{FILE_DIR}/train.csv')['comment_text'].tolist()
test_text = pd.read_csv(f'{FILE_DIR}/test.csv')['comment_text'].tolist()
text_input = train_text + test_text # combine data
text_output = [clean_text(x) for x in text_input] # clean text

# save data
def save_text(file_name, text_array):
    f = open(f'{FILE_DIR}/plain_text/{file_name}.txt', 'w')
    for item in text_array: f.write('%s\n' % item)
        
save_text(f'toxic_comments', text_output)
save_text(f'train_comments', text_output[:159571])
save_text(f'test_comments',  text_output[159571:])

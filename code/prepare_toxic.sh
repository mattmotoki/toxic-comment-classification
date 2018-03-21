#!/bin/bash

# # preprocess data
python preprocessing/clean_text.py
python preprocessing/tokenize_text.py
python preprocessing/tag_pos.py

# # embeddings
./embeddings/create_fasttext_embeddings.sh

# # tfidf features
python tfidf/create_tfidf_features.py 


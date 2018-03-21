import re
import numpy as np
import pandas as pd

def read_embedding(file_name):
  """
  Reads into memory a numpy array of words and a
  corresponding dictionary of numpy word vectors.

  Args:
      file_name: Name of the file to read.

  Returns:
      A tuple of word array and word vector dictionary.
  """

  f = open(file_name, encoding='utf-8')
  word_vectors = {}
  split_pattern = re.compile(u' |\n')
  for line in f:
    split_line = re.split(split_pattern, line.strip())
    word_vectors[split_line[0]] = np.asarray(split_line[1:], dtype='float32')
      
  return word_vectors


def write_embedding(vectors, save_name):
  """
  Saves external word vector to disk.

  Args:
      save_name: Name of the file to save.

  Returns:
      None
  """  

  fwrite = open(save_name, 'w')
  for word, vec in vectors.items():
    fwrite.write(word + ' ' + ' '.join(vec.astype(str)) + '\n')
  fwrite.close()          


def impute_missing(local_vectors, external_vectors, vectorized=False, use_gpu=False, chunk_size=500):
  """
  For each missing word in the external dataset, find the most
  similar word present in both the local and external dataset.

  Args:
    vectorized: If True the missing words are updated one at a time
      in a for loop, otherwise the computationis vectorized.
      (default is True).  Ignored if use_gpu=True.

    use_gpu: If True use gpu to do the computation. (default is False)

    chunk_size: number of words to process on the gpu at once 
    (defaults is 500)

  Returns:
    Dictionary of imputed word vectors.
  """
  import torch
  from sklearn.preprocessing import normalize  

  # find missing words
  local_words, external_words = list(local_vectors.keys()), list(external_vectors.keys())
  shared_words = np.intersect1d(local_words, external_words)
  missing_words = np.setdiff1d(local_words, external_words)

  # create reference matrix
  reference_matrix = np.array([local_vectors[w] for w in shared_words])
  reference_matrix = normalize(reference_matrix).T # word vectors are columns

  # create lookup matrix
  lookup_matrix = np.array([local_vectors[w] for w in missing_words])
  lookup_matrix = normalize(lookup_matrix)

  # perform lookup
  if use_gpu:

    # setup
    n_lookups = lookup_matrix.shape[0]
    n_chunks = n_lookups//chunk_size+1

    # convert to numpy array to torch tensors
    dtype = torch.cuda.FloatTensor  
    def np2tc(x): return torch.from_numpy(x).type(dtype)
    reference_matrix_gpu = np2tc(reference_matrix)
    
    # iterate through chunks
    for i in range(n_chunks):
      chunk_indexs = slice(chunk_size*i, min(chunk_size*(i+1), n_lookups))
      similarity = torch.mm(np2tc(lookup_matrix[chunk_indexs]), reference_matrix_gpu)
      _, similar_indexs = torch.max(similarity, 1)
      similar_words = shared_words[np.array(similar_indexs)]
      for m,s in zip(missing_words[chunk_indexs], similar_words):
        external_vectors[m] = external_vectors[s] 

  else: 

    if vectorized:
      for w in missing_words:
        similarity = np.matmul(local_vectors[w], reference_matrix)
        similar_word = shared_words[np.argmax(similarity)]
        external_vectors[w] = external_vectors[similar_word]

    else: 
      similarity = np.matmul(lookup_matrix, reference_matrix)
      similar_words = shared_words[np.argmax(similarity, axis=1)]
      for m,s in zip(missing_words, similar_words):
        external_vectors[m] = external_vectors[s]

  # keep only local words
  imputed_vectors = local_vectors
  for w in imputed_vectors:
    imputed_vectors[w] = external_vectors[w]
  
  return imputed_vectors

if __name__ == '__main__':

  # # initialize processsor
  # file_path = '~/home/matt/repos/Kaggle-toxic/jigsaw-matt/input/tmp'
  # local_file = file_path+'/fasttext-1000.txt'  
  # external_file  = file_path+'/glove-1000.txt'
  # processor = EmbeddingProcesser(local_file, external_file)

  # local_vectors = processor.local_vectors
  # external_vectors = processor.impute_missing()

  # # create reference matrix
  # reference_matrix = np.array([processor.local_vectors[w] for w in processor.shared_words])
  # reference_matrix = normalize(reference_matrix).T # word vectors are columns

  # # find words similar to random missing words
  # for w in np.random.choice(processor.missing_words, 10):
  #   similarity = np.matmul(local_vectors[w], reference_matrix)
  #   similar_word = processor.shared_words[np.argmax(similarity)]
  #   print('missing: {0: <10}   most similar: {1}'.format(w[:10], similar_word))  
  #   assert np.all(external_vectors[w] == external_vectors[similar_word])

  # # check that the new external set is exhaustive
  # missing = np.setdiff1d(list(local_vectors.keys()), list(external_vectors.keys()))
  # assert len(missing) == 0
  print(0)

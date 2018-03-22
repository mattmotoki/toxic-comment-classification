## 33rd Place Solution Using Embedding Imputation

This is the writeup and code for the [Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge).  

The task was to classify online comments into 6 categories: `toxic`, `severve_toxic`, `obscene`, `threat`, `insult`, `identity_hate`.  The competition metric was the average of the individual AUCs of each predicted class


### Summary of approach:

__Private LB: 0.9872, 33/4551__; Public LB: 0.9876, 45/4551

#### Embeddings: 
   - fastText embeddings trained locally on the competition data
   - Pretrained embeddings (with similiarity imputation): 
      * [fastText: wiki.en.bin](https://fasttext.cc/docs/en/english-vectors.html)
      * [GloVe: glove.840B.300d](https://nlp.stanford.edu/projects/glove/) 
      * [LexVec: lexvec.commoncrawl.300d.W.pos.vectors](https://github.com/alexandres/lexvec)

#### Models (best private score shown): 
   - CapsuleNet    (*0.9860 private*,	0.9859 public)
   - RNN Version 1 (*0.9858 private*,	0.9863 public)
   - RNN Version 2 (*0.9856 private*,	0.9861 public)
   - Two Layer CNN (*0.9826 private*,	0.9835 public)
   - NB-SVM (*0.9813 private*, 0.9813 public)
	
#### Ensembling (best private score shown):
   - Level 1a: Average 10 out-of-fold predictions (as high as *0.9860 private*, 0.9859 public)
   - Level 1b: Average models with different embeddings (as high as *0.9866 private*, 0.9871 public)
   - Level 2a: LightGBM Stacking (*0.9870 private*, 0.9874 public)
   - Level 2b: Average multiple seeds (*0.9872 private*, 0.9876 public)
   
### Embedding Imputation Details:


My main insight in this competition was how to handle out-of-vocabulary (OOV) words.  Replacing missing vectors with zeros or random numbers is suboptimal.  Using fastText's built-in OOV prediction instead of naive replacement increases the AUC by ~0.002.  For GloVe and LexVec embeddings, I replaced the missing embeddings with similar vectors. To do this, I first trained a fastText model on the data for this competition:
```
  fasttext skipgram -input "${INPUT_FILE}" -output "${OUTPUT_FILE}" \
  -minCount 1 -neg 25 -thread 8 -dim 300
```
The `-minCount 1` flag ensures that we get perfect recall; i.e., we get a vector for every word in our vocabulary.  We can now find the most similar vector in the intersection of the local vocabulary (from this competition) with the external vocabulary (from pretrained embeddings).  Here's the psuedo code to do that<sup>[1](#footnote1)</sup>:
```
local = {local_words: local_vectors}
external = {external_words: external_vectors}
shared_words = intersect(local_words, external_words)
missing_words = setdiff(local_words, external_words)
reference_matrix = array(local[w] for w in shared_words).T

for w in missing_words:
     similarity = local[w] * reference_matrix
     most_similar_word = shared_words[argmax(similarity)]
     external[w] = external_vectors[most_similar_word]

return {w: external[w] for w in local_words}
```
With this technique, GloVe performed just as well if not better than the fastText with OOV prediction; LexVec performed slightly worse but added valuable diversity to ensembles. 


#### Timing:
The bulk of the calculation boils down to a vector matrix multiplication.  The naive implementation takes about 20 mins. We can reduce this to about 4 mins by processing missing words in batches.  Using PyTorch (and a 1080ti), we can get the timing down to about 1 min. 

#### Results:
Here is a table of the scores for a single seed; here `toxic` refers to the 300d vectors trained locally using fastText. 


| Model	| Embeddings | Private | Public | Local |
|:------ |:---------- | ------- | ------ | ----- |
|  |
| CapsuleNet	| fasttext	| 0.9855	| 0.9867	| 0.9896|
| CapsuleNet	| glove	| 0.9860 	| 0.9859	| 0.9899|
| CapsuleNet	| lexvec	| 0.9855	| 0.9858	| 0.9898|
| CapsuleNet	| toxic	| 0.9859	| 0.9863	| 0.9901|
|  |
| RNN Version 2	| fasttext	| 0.9856	| 0.9864	| 0.9904|
| RNN Version 2	| glove	| 0.9858 	| 0.9863	| 0.9902|
| RNN Version 2	| lexvec	| 0.9857	| 0.9859	| 0.9902|
| RNN Version 2	| toxic	| 0.9851	| 0.9855	| 0.9906|
|  |
| RNN Version 1	| fasttext	| 0.9853	| 0.9859	| 0.9898|
| RNN Version 1	| glove	| 0.9855	| 0.9861	| 0.9901|
| RNN Version 1	| lexvec	| 0.9854	| 0.9857	| 0.9897|
| RNN Version 1	| toxic	| 0.9856 | 0.9861	| 0.9903|
|  |
| 2 Layer CNN	| fasttext	| 0.9826	| 0.9835	| 0.9886|
| 2 Layer CNN	| glove 	| 0.9827	| 0.9828	| 0.9883|
| 2 Layer CNN	| lexvec	| 0.9824	| 0.9831	| 0.9880|
| 2 Layer CNN	| toxic	| 0.9806	| 0.9789	| 0.9880|
|  |
| SVM with NB features	| NA	| 0.9813	| 0.9813	| 0.9863|

<a name="footnote1"><sup>1</sup></a> This is assuming all word vectors are normalized so that the inner product is the same as the cosine similarity.  

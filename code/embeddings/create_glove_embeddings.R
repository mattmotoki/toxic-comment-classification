library(stringr)
library(text2vec)
library(tokenizers)

# text file reader
loadText <- function(file_name) {
  loadData(file_name, reader=fread, sep='\n', file_ext='txt',
           header=F, blank.lines.skip=T)[[1]]
}

# load data
text <- loadText('toxic_comments')

# tokenize
it <- itoken(text, tokenizer = space_tokenizer)

# Create vocabular, terms will be unigrams
vocab <- create_vocabulary(it) %>% prune_vocabulary(term_count_min=1L)
vectorizer <- vocab_vectorizer(vocab)

# use window of 5 for context words
tcm <- create_tcm(it, vectorizer, skip_grams_window=5L)

# create vectors of length 100, 50
for (n in c(100, 50)) {
  
  # create glove vectors
  glove <- GlobalVectors$new(word_vectors_size=n, vocabulary=vocab, x_max=10)
  wv_matrix <- glove$fit_transform(tcm, n_iter=1000, convergence_tol=1e-6)
  
  # convert to data.table
  wv_table <- data.table(wv_matrix)
  rownames(wv_table) <- rownames(wv_matrix)
  
  # save data
  savename <- sprintf('toxic_glove_%dd',  n)  
  saveData(wv_table, savename, folder='embeddings', subfolder='glove',
           file_ext='txt', writer=fwrite, sep=' ', row.names=T, col.names=F)  
}


## notes
# - toxic_basic_glove_200d, training error
# - toxic.sarcasm.glove200, training error
# - toxic.200, training error
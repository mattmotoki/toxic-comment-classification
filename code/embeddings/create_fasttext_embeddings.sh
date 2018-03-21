#!/bin/bash
DATA_DIR='../input'  

rm_first_line()
{
  FILE_NAME=$1
  tail -n +2 "${FILE_NAME}" > "tmp.txt"
  mv -f "tmp.txt" "${FILE_NAME}"
}

train_word_embeddings()
{
  INPUT_FILE="${DATA_DIR}/plain_text/toxic_comments.txt"
  OUTPUT_FILE="${DATA_DIR}/embeddings/fasttext/toxic_fasttext_300d"

  # train fasttext vectors
  fasttext skipgram -input "${INPUT_FILE}" -output "${OUTPUT_FILE}" \
  -minCount 1 -neg 25 -thread 30 -dim 300

  # raname and remove first line of file
  mv "${OUTPUT_FILE}.vec" "${OUTPUT_FILE}.txt"
  rm_first_line "${OUTPUT_FILE}.txt"
}

train_word_embeddings


train_pos_embeddings()
{
  INPUT_FILE="${DATA_DIR}/part_of_speech/toxic_pos.csv"
  OUTPUT_FILE="${DATA_DIR}/embeddings/fasttext/toxic_pos_fasttext_100d"

  # train fasttext vectors
  fasttext skipgram -input "${INPUT_FILE}" -output "${OUTPUT_FILE}" \
  -minCount 1 -ws 10 -neg 25 -thread 30

  # raname and remove first line of file
  mv "${OUTPUT_FILE}.vec" "${OUTPUT_FILE}.txt"
  rm_first_line "${OUTPUT_FILE}.txt"
}

# train_pos_embeddings


predict_word_embeddings()
{
  MODEL_FILE="${DATA_DIR}/embeddings/fasttext/wiki.en.bin"
  INPUT_FILE="${DATA_DIR}/tokens/toxic_tokens.txt"
  OUTPUT_FILE="${DATA_DIR}/embeddings/fasttext/toxic_lookup_en_wiki_fasttext_300d.txt"
  echo "Lookup fastText Vectors"

  # use model to predict on input tokens
  fasttext print-word-vectors "${MODEL_FILE}" < "${INPUT_FILE}" > "${OUTPUT_FILE}"
  
  # remove the first line of file
  rm_first_line "${OUTPUT_FILE}"
}

# predict_word_embeddings


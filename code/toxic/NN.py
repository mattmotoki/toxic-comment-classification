import time
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, log_loss

from toxic.embedding_utils import read_embedding

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


CLASS_LIST = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# helper function for reading text files
def read_comments(file_name):
  return [x for x in open(file_name, encoding='utf-8')]  

# get random number of the specified type
def get_random(rand_type, lims):

    if rand_type == 'int':
        return int(np.random.randint(lims[0], lims[1]+1, 1)[0])

    if rand_type == 'unif':
        return np.random.uniform(lims[0], lims[1], 1)[0]
        
    if rand_type == 'exp':
        return 10**np.random.uniform(lims[0], lims[1], 1)[0]

# wrapper for competition metric
def avg_auc(Y_true, Y_pred):
    return np.mean([roc_auc_score(Y_true[:,j], Y_pred[:,j]) for j in range(6)])  
    
class NN():

    def __init__(self, model_name, max_seq_len, file_dir,
         word_embedding_file, pos_embedding_file=None):

        self.file_dir = file_dir
        self.word_embedding_file = word_embedding_file
        self.pos_embedding_file = pos_embedding_file
        self.model_name = model_name
        self.max_seq_len = max_seq_len        
        self.Y, self.X, self.X_test = None, None, None
        self.word_embedding_matrix = None
        self.word_embedding_mean = None
        self.word_embedding_std = None
        
    def initialize(self): pass

    def get_model(self): pass

    def get_random_params(self, param_lims):
        
        # specify random type
        param_types = {
            'n_capsule': 'int',
            'capsule_dim': 'int',
            'n_routings': 'int',
            'n_dense': 'int',
            'n_filters': 'int',
            'n_recurrent': 'int',
            'dropout_rate': 'unif',
            'l2_penalty': 'exp'
        }

        # fill in desired parameters with random values
        params = param_lims.copy()
        for k,v in param_lims.items():
            params[k] = get_random(param_types[k], v) if isinstance(v, tuple) else v
        return params

    def load_labels(self):
        print('loading labels')
        train = pd.read_csv(f'{self.file_dir}/train.csv')
        self.Y = train[CLASS_LIST].values

    def load_word_sequences(self):
        print('loading word sequences')
        
        # load text data
        toxic_text = read_comments(f'{self.file_dir}/plain_text/toxic_comments.txt')
        train_text = read_comments(f'{self.file_dir}/plain_text/train_comments.txt')
        test_text  = read_comments(f'{self.file_dir}/plain_text/test_comments.txt')

        # fit tokens
        self.word_tokenizer = Tokenizer()
        self.word_tokenizer.fit_on_texts(toxic_text)

        # convert text to sequences
        self.X = self.text2seq(self.word_tokenizer, train_text)
        self.X_test  = self.text2seq(self.word_tokenizer, test_text)


    def load_word_vectors(self):
        print(f'loading word vectors from {self.word_embedding_file}')
        
        # load word vectors
        word_vectors = read_embedding(f'{self.file_dir}/{self.word_embedding_file}')

        # create matrix of all zeros
        n_words = len(self.word_tokenizer.word_index)+1
        self.word_embedding_matrix = np.zeros((n_words, 300))
        
        # fill in missing embeddings
        self.missing_indexs = []
        for w, i in self.word_tokenizer.word_index.items():
            if w in word_vectors: 
                self.word_embedding_matrix[i] = word_vectors[w]
            else: 
                self.missing_indexs.append(i)

        self.missing_indexs = np.array(self.missing_indexs)


    def text2seq(self, tokenizer, text):
        """Convert text to an nd-array of sequences"""
        tokens = tokenizer.texts_to_sequences(text)   
        seq = pad_sequences(tokens, maxlen=self.max_seq_len)
        return seq

    def predict_kfold(self, param_lims, n_folds=3, seed=0,
         monitor_training=True, save_oof=False):
        """
        Perform k-fold cross validation with (possibly) random hyperparameters.
        Training and validation output is logged and stored in a log file.  Test
        set predictions are made and saved on every fold.  After the each full 
        run, the test predictions are aggregated using the harmonic average.

        Args:
            param_lims: A dictionary of hyperparameter limits.  Keys are the parameter
                name and values can be either a single value (no randomization) or a 
                tuple specifying `(min_val, max_val)`.
            n_folds: number of CV folds (default=3)
            seed: random seed (default=0)
            monitor_training: calculate loss and auc on the training set (default=True)
            save_oof: save out-of-fold-predictions (default=False)
        """

        # create storage containers
        oof_preds, oof_scores, progress_log = np.zeros(self.Y.shape), [], []
        test_preds = [np.zeros((self.X_test.shape[0], 6)) for k in range(n_folds)]
        
        # parse params
        params = self.get_random_params(param_lims)
        param_message = '  '.join([f'{k}:{v:0.5g}' for k,v in params.items()]) 
        progress_log.append(param_message)
        print(param_message)
            
        # print model summary
        model = self.get_model(**params)
        model.summary(print_fn=lambda x: progress_log.append(x))
        print(model.summary())
        
        # iterate through folds
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for k, (train_index, valid_index) in enumerate(kf.split(self.X)):

            start_message = f'\ntraining on iteration {k+1} out of {n_folds}'
            progress_log.append(start_message)
            print(start_message)    
            
            # create name for files
            full_model_name = f'{self.model_name}_{k+1}of{n_folds}_seed{seed}'
            weight_file = f'{self.file_dir}/model_weights/{full_model_name}.h5py'

            # unpack data
            X_train, Y_train = self.X[train_index], self.Y[train_index]
            X_valid, Y_valid = self.X[valid_index], self.Y[valid_index]

            # get new model
            model = self.get_model(**params)

            # train with early stopping        
            it, count, valid_auc, best_auc = 0, 0, 0, 0
            while count<2:
                it += 1

                # fit model
                start_time = time.time()
                model.fit(
                    X_train, Y_train, 
                    validation_data=(X_valid, Y_valid),
                    batch_size=256, epochs=1, verbose=0)
                end_time = time.time()

                # predict and evalute
                train_preds = model.predict(X_train) if monitor_training else -1
                valid_preds = model.predict(X_valid)
                
                train_loss = log_loss(Y_train.flatten(), train_preds.flatten()) if monitor_training else -1
                valid_loss = log_loss(Y_valid.flatten(), valid_preds.flatten())

                train_auc = avg_auc(Y_train, train_preds) if monitor_training else -1
                valid_auc = avg_auc(Y_valid, valid_preds)
                
                # update early stopping
                if (valid_auc > best_auc):
                    model.save_weights(weight_file)
                    oof_preds[valid_index] = valid_preds
                    best_auc = valid_auc
                    count = 0
                else:
                    count += 1

                # log progress
                progress = f'  {it:02}  '\
                f'train_loss: {train_loss:.5f}  valid_loss: {valid_loss:.5f}   '\
                f'train_auc: {train_auc:.5f}  valid_auc: {valid_auc:.5f}   '\
                f'time: {(end_time - start_time)/60:02.2f} mins, {(time.time()-end_time)/60:02.2f} mins    '
                progress_log.append(progress)
                print(progress)
            
            # make predictions on the test set
            model.load_weights(weight_file)
            test_preds[k] = model.predict(self.X_test)
            oof_scores.append(best_auc)

            # save single fold predictions
            subm = pd.read_csv(f'{self.file_dir}/sample_submission.csv')
            for j, class_name in enumerate(CLASS_LIST):
                subm[class_name] = test_preds[k][:,j]
            
            save_name = f'{self.file_dir}/submissions/singles/{full_model_name}-{best_auc:.5f}.csv'
            subm.to_csv(save_name, index=False)


        # log overall progress
        progress = f'\nmean_auc={np.mean(oof_scores):.5f} best_auc={np.max(oof_scores):.5f}'
        progress_log.append(progress)
        print(progress)

        # save log
        save_name = f'{self.file_dir}/logs/{self.model_name}_'\
        f'{n_folds}folds_seed{seed}-{np.mean(oof_scores):.5f}.txt'
        fwrite = open(save_name, 'w') 
        for line in progress_log: 
            fwrite.write(f'{line}\n') 
        fwrite.close()     

        # save out-of-fold predictions
        if save_oof:
            oof_preds = pd.DataFrame(oof_preds, columns=CLASS_LIST)
            save_name_valid = f'{self.file_dir}/out_of_fold_preds/'\
            f'{self.model_name}_validation_seed{seed}-{np.mean(oof_scores):.5f}.csv'
            oof_preds.to_csv(save_name_valid, index = False)


        # aggregate test predictions
        hmean_test_preds = np.zeros(test_preds[0].shape)
        for i in range(n_folds):
            hmean_test_preds += 1/test_preds[i]
        hmean_test_preds = n_folds/hmean_test_preds

        subm = pd.read_csv(f'{self.file_dir}/sample_submission.csv')
        for j, class_name in enumerate(CLASS_LIST):
            subm[class_name] = hmean_test_preds[:,j]
            
        # save aggregate predictions
        save_name = f'{self.file_dir}/submissions/aggregate/{self.model_name}_'\
        f'avgof{n_folds}_seed{seed}-{np.mean(oof_scores):.5f}.csv'
        subm.to_csv(save_name, index=False)    

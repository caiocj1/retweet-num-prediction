import gensim.models
import pandas as pd
import numpy as np
import os
import yaml
import ast
from yaml import SafeLoader
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from typing import Optional
from collections import defaultdict

class RetweetDataModule(LightningDataModule):
    def __init__(self,
            split_seed: int = 12345,  # split needs to be always the same for correct cross validation
            num_splits: int = 10,
            batch_size: int = 32,
            num_workers: int = 0,
            max_samples: int = None):
        super().__init__()
        dataset_path = os.getenv('DATASET_PATH')
        self.train_path = os.path.join(dataset_path, 'train.csv')
        self.test_path = os.path.join(dataset_path, 'evaluation.csv')

        # Save hyperparemeters
        self.save_hyperparameters(logger=False)

        # Read config file
        self.read_config()

        # Declare data
        self.data_train = None
        self.data_val = None
        self.data_predict = None

        # Prepare split
        self.kf = KFold(n_splits=self.hparams.num_splits, shuffle=True, random_state=self.hparams.split_seed)

        # Get training set
        self.train_df = pd.read_csv(self.train_path)
        if max_samples is not None:
            self.train_df = self.train_df.iloc[:max_samples]

        self.train_df_input = self.train_df.drop(['TweetID', 'timestamp', 'mentions', 'retweets_count', 'text'], axis=1)

        self.train_df_input.urls = self.train_df_input.urls.apply(ast.literal_eval)
        self.train_df_input.urls = self.train_df_input.urls.apply(len)

        self.train_df_input.hashtags = self.train_df_input.hashtags.apply(ast.literal_eval)
        self.train_df_input.hashtags = self.train_df_input.hashtags.apply(len)

        self.train_mean = self.train_df_input.values.mean(0)
        self.train_std = self.train_df_input.values.std(0)

        # Load word2vec
        if self.apply_w2v:
            self.word2vec = gensim.models.Word2Vec.load('models/word2vec.model')

            train_tweets = self.train_df['text'].apply(str.split).to_list()
            self.train_dictionary = gensim.corpora.Dictionary(train_tweets)
            self.train_corpus = [self.train_dictionary.doc2bow(tweet) for tweet in train_tweets]

            self.tfidf = gensim.models.TfidfModel(self.train_corpus)

        # Fit eventual PCA
        if self.apply_pca:
            self.pca = PCA(self.reduced_dims)
            train_df_normalized = (self.train_df_input.values - self.train_mean) / self.train_std
            self.pca.fit(train_df_normalized)

    def read_config(self):
        config_path = os.path.join(os.getcwd(), 'config.yaml')
        with open(config_path) as f:
            params = yaml.load(f, Loader=SafeLoader)
        dataset_params = params['DatasetParams']

        self.apply_w2v = dataset_params['apply_w2v']
        self.apply_pca = dataset_params['apply_pca']
        self.reduced_dims = dataset_params['reduced_dims']

    def setup(self, stage: str = None, k: int = 0):
        assert 0 <= k < self.hparams.num_splits, "incorrect fold number"

        if stage == 'fit':
            # Choose fold to train on
            all_splits = [i for i in self.kf.split(self.train_df_input)]
            train_indexes, val_indexes = all_splits[k]

            train_df = self.train_df_input.iloc[train_indexes]
            val_df = self.train_df_input.iloc[val_indexes]

            # Get inputs
            train_X = (train_df.values - self.train_mean) / self.train_std
            val_X = (val_df.values - self.train_mean) / self.train_std

            if hasattr(self, 'pca'):
                train_X = self.pca.transform(train_X)
                val_X = self.pca.transform(val_X)

            train_X = dict(enumerate(train_X))
            val_X = dict(enumerate(val_X))

            # Get labels
            train_y = dict(enumerate(self.train_df.iloc[train_indexes]['retweets_count'].values))
            val_y = dict(enumerate(self.train_df.iloc[val_indexes]['retweets_count'].values))

            # Get dict
            train_dict = defaultdict()
            val_dict = defaultdict()
            train_text = self.train_df['text'].iloc[train_indexes]
            val_text = self.train_df['text'].iloc[val_indexes]
            for i in range(len(train_y)):
                if hasattr(self, 'word2vec'):
                    keys = [self.train_dictionary.token2id[word] for word in train_text.iloc[i].split(' ')]
                    tf_idf_dict = dict(self.tfidf[self.train_corpus[train_indexes[i]]])
                    tf_idf_coefs = np.array([tf_idf_dict[key] for key in keys])

                    text_vec = self.word2vec.wv[train_text.iloc[i].split(' ')]
                    text_vec = (text_vec * tf_idf_coefs[:, None]).sum(0)
                    train_X[i] = np.concatenate([train_X[i], text_vec])
                train_dict[i] = (train_X[i], train_y[i])
            for i in range(len(val_y)):
                if hasattr(self, 'word2vec'):
                    keys = [self.train_dictionary.token2id[word] for word in val_text.iloc[i].split(' ')]
                    tf_idf_dict = dict(self.tfidf[self.train_corpus[val_indexes[i]]])
                    tf_idf_coefs = np.array([tf_idf_dict[key] for key in keys])

                    text_vec = self.word2vec.wv[val_text.iloc[i].split(' ')]
                    text_vec = (text_vec * tf_idf_coefs[:, None]).sum(0)
                    val_X[i] = np.concatenate([val_X[i], text_vec])
                val_dict[i] = (val_X[i], val_y[i])

            self.data_train, self.data_val = train_dict, val_dict

        elif stage == 'predict':
            test_full = pd.read_csv(self.test_path)
            self.test_ids = test_full['TweetID']

            test_text = test_full['text']

            test_tweets = test_text.apply(str.split).to_list()
            test_corpus = [self.train_dictionary.doc2bow(tweet) for tweet in test_tweets]

            test_full = test_full.drop(['TweetID', 'timestamp', 'mentions', 'text'], axis=1)

            test_full.urls = test_full.urls.apply(ast.literal_eval)
            test_full.urls = test_full.urls.apply(len)

            test_full.hashtags = test_full.hashtags.apply(ast.literal_eval)
            test_full.hashtags = test_full.hashtags.apply(len)

            test_X = (test_full.values - self.train_mean) / self.train_std
            if hasattr(self, 'pca'):
                test_X = self.pca.transform(test_X)

            test_X = dict(enumerate(test_X))

            test_dict = defaultdict()
            for i in range(len(test_X)):
                if hasattr(self, 'word2vec'):
                    encoded_words = [word for word in test_text.iloc[i].split(' ') if word in self.word2vec.wv]
                    if encoded_words:
                        keys = [self.train_dictionary.token2id[word] for word in encoded_words]
                        tf_idf_dict = dict(self.tfidf[test_corpus[i]])
                        tf_idf_coefs = np.array([tf_idf_dict[key] for key in keys])

                        text_vec = self.word2vec.wv[test_text.iloc[i].split(' ')]
                        text_vec = (text_vec * tf_idf_coefs[:, None]).sum(0)
                        test_X[i] = np.concatenate([test_X[i], text_vec])
                    else:
                        test_X[i] = np.concatenate([test_X[i], np.zeros((self.word2vec.vector_size,))])
                test_dict[i] = (test_X[i], -1)

            self.data_predict = test_dict

    def train_dataloader(self):
        return DataLoader(dataset=self.data_train,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(dataset=self.data_val,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers,
                          shuffle=False)

    def predict_dataloader(self):
        return DataLoader(dataset=self.data_predict,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers,
                          shuffle=False)
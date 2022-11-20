import gensim.models
import pandas as pd
import numpy as np
import os
import yaml
import ast
import datetime
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
        read_train_df = pd.read_csv(self.train_path)
        if max_samples is not None:
            read_train_df = read_train_df.iloc[:max_samples]

        temp = read_train_df.explode('urls')[['urls', 'retweets_count']]
        self.avg_per_url = temp.groupby(['urls']).mean().to_dict()['retweets_count']

        self.train_df = self.format_df(read_train_df, keep_time=self.keep_time, keep_fts=True)

        self.train_df_input = self.train_df.drop(['retweets_count', 'text'], axis=1)

        self.train_mean = self.train_df_input.values.mean(0)
        self.train_std = self.train_df_input.values.std(0)

        # Get test set
        self.test_df = pd.read_csv(self.test_path)
        self.test_ids = self.test_df['TweetID']
        self.test_df = self.format_df(self.test_df, type='test', keep_time=self.keep_time, keep_fts=True)

        self.test_df_input = self.test_df.drop(['text'], axis=1)

        # Load word2vec
        if self.apply_w2v:
            self.word2vec = gensim.models.Word2Vec.load('models/word2vec.model')

            train_tweets = self.train_df['text'].apply(str.split).to_list()
            self.train_dictionary = gensim.corpora.Dictionary(train_tweets)
            self.train_corpus = [self.train_dictionary.doc2bow(tweet) for tweet in train_tweets]

            test_tweets = self.test_df['text'].apply(str.split).to_list()
            self.test_corpus = [self.train_dictionary.doc2bow(tweet) for tweet in test_tweets]

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

        self.keep_time = dataset_params['keep_time']
        self.apply_w2v = dataset_params['apply_w2v']
        self.apply_pca = dataset_params['apply_pca']
        self.reduced_dims = dataset_params['reduced_dims']

    def setup(self, stage: str = None, k: int = 0):
        assert 0 <= k < self.hparams.num_splits, "incorrect fold number"

        if stage == 'fit':
            # Choose fold to train on
            all_splits = [i for i in self.kf.split(self.train_df_input)]
            train_indexes, val_indexes = all_splits[k]

            train_dict = self.format_X(self.train_df_input.iloc[train_indexes],
                                       self.train_mean,
                                       self.train_std,
                                       type='train',
                                       indexes=train_indexes)
            val_dict = self.format_X(self.train_df_input.iloc[val_indexes],
                                     self.train_mean,
                                     self.train_std,
                                     type='train',
                                     indexes=val_indexes)

            self.data_train, self.data_val = train_dict, val_dict

        elif stage == 'predict':
            predict_dict = self.format_X(self.test_df_input,
                                         self.train_mean,
                                         self.train_std,
                                         type='test')

            self.data_predict = predict_dict

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

    def format_df(self,
                  df: pd.DataFrame,
                  type: str = 'train',
                  keep_time: bool = False,
                  keep_fts: bool = False):
        final_df = df.drop(['TweetID', 'mentions', 'timestamp'], axis=1)

        final_df.urls = final_df.urls.apply(ast.literal_eval)
        final_df.hashtags = final_df.hashtags.apply(ast.literal_eval)

        if keep_fts:
            def apply_avg_urls(list_obj):
                sum = 0
                for url in list_obj:
                    sum += self.avg_per_url[url] if url in self.avg_per_url else 0.0
                return sum / len(list_obj) if len(list_obj) > 0 else 0.0
            url_avg_retweets = df['urls'].apply(apply_avg_urls)
            final_df = pd.concat([final_df, url_avg_retweets], axis=1)

            text_len = df['text'].apply(len)
            final_df = pd.concat([final_df, text_len], axis=1)

        final_df.urls = final_df.urls.apply(len)
        final_df.hashtags = final_df.hashtags.apply(len)

        if keep_time:
            timestamps = df.timestamp // 1000
            timestamps = timestamps.apply(datetime.datetime.fromtimestamp).apply(datetime.datetime.timetuple)

            time_df = pd.DataFrame(timestamps.tolist(), index=df.index,
                                   columns=['tm_year', 'tm_mon', 'tm_mday', 'tm_hour', 'tm_min', 'tm_sec', 'tm_wday',
                                            'tm_yday', 'tm_isdst'])
            time_df = time_df.drop(['tm_year', 'tm_mon', 'tm_mday', 'tm_isdst'], axis=1)

            final_df = pd.concat([final_df, time_df], axis=1)

        return final_df

    def format_X(self,
                 df: pd.DataFrame,
                 mean: np.ndarray,
                 std: np.ndarray,
                 type: str = 'train',
                 indexes: np.ndarray = None):
        # Get inputs
        X = (df.values - mean) / std

        if hasattr(self, 'pca'):
            X = self.pca.transform(X)

        X = dict(enumerate(X))

        # Get labels
        if type == 'train':
            y = dict(enumerate(self.train_df.iloc[indexes]['retweets_count'].values))
        else:
            y = dict(enumerate(np.zeros((len(X), ))))

        # Get dict
        final_dict = defaultdict()

        if type == 'train':
            text = self.train_df['text'].iloc[indexes]
        else:
            text = self.test_df['text']
        for i in range(len(y)):
            if hasattr(self, 'word2vec'):
                encoded_words = [word for word in text.iloc[i].split(' ') if word in self.word2vec.wv and
                                 word in self.train_dictionary.token2id]
                if encoded_words:
                    keys = [self.train_dictionary.token2id[word] for word in encoded_words]
                    if type == 'train':
                        tf_idf_dict = dict(self.tfidf[self.train_corpus[indexes[i]]])
                    else:
                        tf_idf_dict = dict(self.tfidf[self.test_corpus[i]])
                    tf_idf_coefs = np.array([tf_idf_dict[key] for key in keys])

                    text_vec = self.word2vec.wv[encoded_words]
                    text_vec = (text_vec * tf_idf_coefs[:, None]).sum(0)
                    X[i] = np.concatenate([X[i], text_vec])
                else:
                    X[i] = np.concatenate([X[i], np.zeros((self.word2vec.vector_size,))])
            final_dict[i] = (X[i], y[i])

        return final_dict
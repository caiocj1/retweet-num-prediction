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
import sklearn
import sklearn.cluster
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

        # Prepare split
        self.kf = KFold(n_splits=self.hparams.num_splits, shuffle=True, random_state=self.hparams.split_seed)


        # Get training set
        read_train_df = pd.read_csv(self.train_path)
        read_train_df.urls = read_train_df.urls.apply(ast.literal_eval)
        read_train_df.hashtags = read_train_df.hashtags.apply(ast.literal_eval)
        if max_samples is not None:
            read_train_df = read_train_df.iloc[:max_samples]

        self.train_df = self.format_df(read_train_df)

        self.train_df_input = self.feature_engineering(self.train_df, type='train')

        self.train_mean = self.train_df_input.values.mean(0)
        self.train_std = self.train_df_input.values.std(0)


        # Get test set
        read_test_df = pd.read_csv(self.test_path)
        read_test_df.urls = read_test_df.urls.apply(ast.literal_eval)
        read_test_df.hashtags = read_test_df.hashtags.apply(ast.literal_eval)
        self.test_ids = read_test_df['TweetID']

        self.test_df = self.format_df(read_test_df, type='test')

        self.test_df_input = self.feature_engineering(self.test_df, type='test')


        # Load word2vec
        if self.apply_w2v:
            self.word2vec = gensim.models.Word2Vec.load('models/word2vec.model')

            train_tweets = self.train_df['text'].to_list()
            self.train_dictionary = gensim.corpora.Dictionary(train_tweets)
            self.train_corpus = [self.train_dictionary.doc2bow(tweet) for tweet in train_tweets]

            test_tweets = self.test_df['text'].to_list()
            self.test_corpus = [self.train_dictionary.doc2bow(tweet) for tweet in test_tweets]

            self.tfidf = gensim.models.TfidfModel(self.train_corpus)

    def read_config(self):
        """
        Read configuration file with hyperparameters.
        :return: None
        """
        config_path = os.path.join(os.getcwd(), 'config.yaml')
        with open(config_path) as f:
            params = yaml.load(f, Loader=SafeLoader)
        word2vec_params = params['Word2VecParams']
        dataset_params = params['DatasetParams']

        self.vector_size = word2vec_params['vector_size']
        self.urls_hashtags_in_text = word2vec_params['urls_hashtags_in_text']

        self.apply_w2v = dataset_params['apply_w2v']

    def setup(self, stage: str = None, k: int = 0):
        """
        Build data dictionaries for training or prediction.
        :param stage: 'fit' for training, 'predict' for prediction
        :param k: which fold to train on
        :return: None
        """
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
        """
        Uses train dictionary (output of format_X) to return train DataLoader, that will be fed to pytorch lightning's
        Trainer.
        :return: train DataLoader
        """
        return DataLoader(dataset=self.data_train,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers,
                          shuffle=True)

    def val_dataloader(self):
        """
        Uses val dictionary (output of format_X) to return val DataLoader, that will be fed to pytorch lightning's
        Trainer.
        :return: train DataLoader
        """
        return DataLoader(dataset=self.data_val,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers,
                          shuffle=False)

    def predict_dataloader(self):
        """
        Uses predict dictionary (output of format_X) to return predict DataLoader, that will be fed to pytorch
        lightning's Trainer.
        :return: predict DataLoader
        """
        return DataLoader(dataset=self.data_predict,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers,
                          shuffle=False)

    def format_df(self,
                  df: pd.DataFrame,
                  type: str = 'train'):
        """
        Formats the .csv read into an initial dataframe with base features.
        :param df: dataframe read from a csv file
        :param type: whether we are treating a training or test set
        :return: correctly formatted dataframe
        """
        final_df = df.drop(['TweetID', 'mentions', 'timestamp'], axis=1)

        final_df['text'] = final_df['text'].apply(str.split)

        if self.urls_hashtags_in_text:
            final_df['text'] = final_df[['text', 'hashtags']].sum(axis=1)

        final_df['url_count'] = final_df.urls.apply(len)
        final_df['hashtag_count'] = final_df.hashtags.apply(len)

        # Parse timestamps
        timestamps = df.timestamp // 1000
        timestamps = timestamps.apply(datetime.datetime.fromtimestamp).apply(datetime.datetime.timetuple)

        time_df = pd.DataFrame(timestamps.tolist(), index=df.index,
                               columns=['tm_year', 'tm_mon', 'tm_mday', 'tm_hour', 'tm_min', 'tm_sec', 'tm_wday',
                                        'tm_yday', 'tm_isdst'])

        final_df = pd.concat([final_df, time_df], axis=1)

        return final_df

    def feature_engineering(self,
                            df: pd.DataFrame,
                            type: str = 'train'):
        """
        Adds non-trivial features to dataframe, drops target/text/useless columns, so as to prepare the input to the
        deep learning model.
        :param df: formatted dataframe from format_df
        :param type: whether we are treating a training or test set
        :return: correctly formatted input dataframe
        """
        final_df = df.drop(['text', 'urls', 'hashtags', 'tm_sec', 'tm_isdst'], axis=1)
        if type == 'train':
            final_df = final_df.drop('retweets_count', axis=1)

        # Does text contain 'rt'?
        def has_rt(word_list):
            return int('rt' in word_list)
        final_df['has_rt'] = df['text'].apply(has_rt)

        # Log features
        final_df['favorites_count_log'] = np.log10(final_df['favorites_count']).replace([-np.inf], -1)
        final_df['followers_count_log'] = np.log10(final_df['followers_count']).replace([-np.inf], -1)
        final_df['statuses_count_log'] = np.log10(final_df['statuses_count']).replace([-np.inf], -1)
        final_df['friends_count_log'] = np.log10(final_df['friends_count']).replace([-np.inf], -1)

        return final_df

    def format_X(self,
                 df: pd.DataFrame,
                 mean: np.ndarray,
                 std: np.ndarray,
                 type: str = 'train',
                 indexes: np.ndarray = None):
        """
        Prepares a dictionary in which to each key is associated a tuple of (input vector, retweet count ground truth),
        from the rows of the dataframe given.
        :param df: correctly formatted input dataframe from feature_engineering
        :param mean: vector with which we normalize the data
        :param std: vector with which we normalize the data
        :param type: whether we are treating a training or test set
        :param indexes: if treating training set, separate train and validation
        :return: correctly dictionary to be passed to DataLoader
        """
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
                encoded_words = [word for word in text.iloc[i] if word in self.word2vec.wv and
                                 word in self.train_dictionary.token2id]
                if encoded_words:
                    keys = [self.train_dictionary.token2id[word] for word in encoded_words]
                    if type == 'train':
                        tf_idf_dict = dict(self.tfidf[self.train_corpus[indexes[i]]])
                    else:
                        tf_idf_dict = dict(self.tfidf[self.test_corpus[i]])
                    tf_idf_coefs = np.array([tf_idf_dict[key] for key in keys])

                    word_matrix = self.word2vec.wv[encoded_words]
                    text_vec = (word_matrix * tf_idf_coefs[:, None]).sum(0)

                    X[i] = np.concatenate([X[i], text_vec])
                else:
                    X[i] = np.concatenate([X[i], np.zeros((self.word2vec.vector_size,))])

            final_dict[i] = (X[i], y[i])

        return final_dict
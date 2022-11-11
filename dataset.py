import pandas as pd
import os
import yaml
from yaml import SafeLoader
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from typing import Optional
from collections import defaultdict

class ForestCoverDataModule(LightningDataModule):
    def __init__(self,
            split_seed: int = 12345,  # split needs to be always the same for correct cross validation
            num_splits: int = 10,
            batch_size: int = 32,
            num_workers: int = 0):
        super().__init__()
        dataset_path = os.getenv('DATASET_PATH')
        self.train_path = os.path.join(dataset_path, 'train.csv')
        self.test_path = os.path.join(dataset_path, 'test-full.csv')

        # Save hyperparemeters
        self.save_hyperparameters(logger=False)

        # Read config file
        self.read_config()

        # Declare data
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None

        # Prepare split
        self.kf = KFold(n_splits=self.hparams.num_splits, shuffle=True, random_state=self.hparams.split_seed)

        # Get training set
        self.train_df = pd.read_csv(self.train_path)
        self.train_df_input = self.train_df.drop(['Id', 'Soil_Type15', 'Cover_Type'], axis=1)
        self.train_mean = self.train_df_input.values.mean(0)
        self.train_std = self.train_df_input.values.std(0)

        # Fit eventual PCA
        if self.reduced_dims:
            self.pca = PCA(self.reduced_dims)
            train_df_normalized = (self.train_df_input.values - self.train_mean) / self.train_std
            self.pca.fit(train_df_normalized)

    def read_config(self):
        config_path = os.path.join(os.getcwd(), 'config.yaml')
        with open(config_path) as f:
            params = yaml.load(f, Loader=SafeLoader)
        dataset_params = params['DatasetParams']

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
            train_y = dict(enumerate(self.train_df.iloc[train_indexes]['Cover_Type'].values - 1))
            val_y = dict(enumerate(self.train_df.iloc[val_indexes]['Cover_Type'].values - 1))

            # Get dict
            train_dict = defaultdict()
            val_dict = defaultdict()
            for i in range(len(train_y)):
                train_dict[i] = (train_X[i], train_y[i])
            for i in range(len(val_y)):
                val_dict[i] = (val_X[i], val_y[i])

            self.data_train, self.data_val = train_dict, val_dict

        elif stage == 'predict':
            test_full = pd.read_csv(self.test_path)
            self.test_ids = test_full['Id']
            test_full = test_full.drop(['Id', 'Soil_Type15'], axis=1)

            test_X = (test_full.values - self.train_mean) / self.train_std
            if hasattr(self, 'pca'):
                test_X = self.pca.transform(test_X)

            test_X = dict(enumerate(test_X))

            test_dict = defaultdict()
            for i in range(len(test_X)):
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
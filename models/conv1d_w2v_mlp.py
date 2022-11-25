import torch.optim
from pytorch_lightning import LightningModule
import torch.nn as nn
import os
import yaml
from yaml import SafeLoader
from collections import OrderedDict
import gensim

class ConvWord2VecModel(LightningModule):

    def __init__(self):
        super(ConvWord2VecModel, self).__init__()
        self.read_config()

        self.build_model()

    def read_config(self):
        config_path = os.path.join(os.getcwd(), './config.yaml')
        with open(config_path) as f:
            params = yaml.load(f, Loader=SafeLoader)
        word2vec_params = params['Word2VecParams']
        dataset_params = params['DatasetParams']
        model_params = params['ModelParams']

        self.vector_size = word2vec_params['vector_size']

        self.keep_time = dataset_params['keep_time']
        self.apply_w2v = dataset_params['apply_w2v']
        self.apply_pca = dataset_params['apply_pca']
        self.reduced_dims = dataset_params['reduced_dims']

        self.layer_width = model_params['layer_width']
        self.num_layers = model_params['num_layers']
        self.dropout = model_params['dropout']

    def build_model(self):
        assert not self.apply_pca, 'Turn off PCA'
        assert self.apply_w2v, 'Turn on Word2Vec'

        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, 5),
            nn.BatchNorm1d(16),
            nn.ReLU(),

            nn.MaxPool1d(2),
            nn.Dropout(p=self.dropout),

            nn.Conv1d(16, 16, 5, stride=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),

            nn.Dropout(p=self.dropout),

            nn.Conv1d(16, 64, 5),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.MaxPool1d(2),
            nn.Dropout(p=self.dropout),

            nn.Conv1d(64, 64, 5, stride=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Dropout(p=self.dropout),

            nn.Conv1d(64, 128, 5),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.MaxPool1d(2),
            nn.Dropout(p=self.dropout),

            nn.Conv1d(128, 128, 3),
            nn.BatchNorm1d(128),
            # new ----------------------
            nn.ReLU(),

            nn.Conv1d(128, 16, 1),
            nn.BatchNorm1d(16),
            # --------------------------
            nn.Flatten()
        )

        input_width = 14 if self.keep_time else 7
        #self.input = nn.Linear(input_width + 256, self.layer_width)
        self.input = nn.Linear(input_width + 32, self.layer_width)

        hidden_layers_dict = OrderedDict()
        for i in range(self.num_layers - 2):
            hidden_layers_dict['layer' + str(i + 1)] = nn.Linear(self.layer_width, self.layer_width)
            hidden_layers_dict['relu' + str(i + 1)] = nn.ReLU()
            hidden_layers_dict['dropout' + str(i + 1)] = nn.Dropout(p=self.dropout)
        self.hidden_layers = nn.Sequential(hidden_layers_dict)

        self.output = nn.Linear(self.layer_width, 1)
        self.relu = nn.ReLU()

    def training_step(self, batch, batch_idx):
        loss, metrics = self._shared_step(batch)

        loss = loss.mean()
        self.log_metrics(metrics, 'train')
        self.log('loss_train', loss, on_step=True, on_epoch=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, metrics = self._shared_step(batch)

        loss = loss.mean()
        self.log_metrics(metrics, 'val')
        self.log('loss_val', loss, on_step=False, on_epoch=True, logger=True)

        return loss

    def test_step(self, batch, batch_idx):
        loss, metrics = self._shared_step(batch)

        loss = loss.mean()
        return loss

    def _shared_step(self, batch):
        prediction = self.forward(batch)

        loss = self.calc_loss(prediction, batch[1])

        metrics = self.calc_metrics(prediction, batch[1])

        return loss, metrics

    def forward(self, batch):
        text_vec = batch[0][:, None, -self.vector_size:].float()
        text_enc = self.conv(text_vec)

        input = torch.concat([batch[0][:, :-self.vector_size].float(), text_enc], dim=1)
        encoding = self.input(input)

        encoding = self.hidden_layers(self.relu(encoding))

        prediction = torch.squeeze(self.output(encoding))

        return prediction

    def calc_loss(self, prediction, target):
        loss_func = nn.L1Loss(reduction='none')

        loss = loss_func(prediction.float(), target.float())

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        #lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)
        #lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=64)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=72, eta_min=5e-5)

        return [optimizer], [lr_scheduler]

    def calc_metrics(self, prediction, target):
        metrics = {}

        metrics['mae'] = torch.abs(prediction - target).mean()

        return metrics

    def log_metrics(self, metrics: dict, type: str):
        on_step = True if type == 'train' else False

        for key in metrics:
            self.log(key + '_' + type, metrics[key], on_step=on_step, on_epoch=True, logger=True)

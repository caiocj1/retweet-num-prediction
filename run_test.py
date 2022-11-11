import argparse
import os

import pandas as pd
import yaml

from model import ForestCoverModel
from dataset import ForestCoverDataModule

import torch
import torch.cuda

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', '-v')
    parser.add_argument('--weights_path', '-w', required=True)

    args = parser.parse_args()

    # Read config file
    config_path = os.path.join(os.getcwd(), 'config.yaml')
    with open(config_path) as f:
        params = yaml.load(f, Loader=yaml.SafeLoader)

    # Model selection
    model = ForestCoverModel()

    data_module = ForestCoverDataModule(
        batch_size=32,
        num_workers=8
    )

    trainer = Trainer(accelerator='auto',
                      devices=1 if torch.cuda.is_available() else None)

    results = []
    for ckpt_name in os.listdir(args.weights_path):
        ckpt_path = os.path.join(args.weights_path, ckpt_name)

        # data_module.prepare_data()
        data_module.setup(stage='predict')

        test_results = trainer.predict(model, data_module, ckpt_path=ckpt_path, return_predictions=True)

        test_results_df = pd.DataFrame(data={'Cover_Type': torch.argmax(torch.cat(test_results), dim=1).numpy() + 1})
        test_ids = pd.DataFrame(data_module.test_ids)

        submission = pd.concat([test_ids, test_results_df], axis=1)
        dataset_path = os.getenv('DATASET_PATH')
        submission_path = os.path.join(dataset_path, 'submission.csv')
        submission.to_csv(submission_path, index=False)

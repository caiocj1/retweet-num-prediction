import argparse
import os
import yaml

from models.no_text_mlp import NoTextMLPModel
from models.w2v_mlp import Word2VecMLPModel
from models.conv1d_w2v_mlp import ConvWord2VecModel
from models.conv2d_w2v_mlp import Conv2DWord2VecModel
from dataset import RetweetDataModule

import torch.cuda

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', '-v')
    parser.add_argument('--model', '-m', default='NoTextMLP')
    parser.add_argument('--weights_path', '-w', default=None)

    args = parser.parse_args()

    # Read config file
    config_path = os.path.join(os.getcwd(), 'config.yaml')
    with open(config_path) as f:
        params = yaml.load(f, Loader=yaml.SafeLoader)
    training_params = params['TrainingParams']

    # Initialize data module
    data_module = RetweetDataModule(
        split_seed=training_params['split_seed'],
        num_splits=training_params['num_splits'],
        batch_size=32,
        num_workers=0,
        max_samples=None)

    for k in range(training_params['num_splits']):
        print('Training on split', k, '...')

        # Initialize new model and setup data module
        model = None
        if args.model == 'mlp':
            model = NoTextMLPModel()
        elif args.model == 'w2v':
            model = Word2VecMLPModel()
        elif args.model == 'conv':
            model = ConvWord2VecModel()
        elif args.model == 'conv2d':
            model = Conv2DWord2VecModel()

        if args.weights_path is not None:
            model = model.load_from_checkpoint(args.weights_path)

        data_module.setup(stage='fit', k=k)

        # Loggers and checkpoints
        version = args.version + '_split=' + str(k)
        logger = TensorBoardLogger('.', version=version)
        model_ckpt = ModelCheckpoint(dirpath=f'lightning_logs/{args.version}_CV/checkpoints',
                                     filename='{epoch}-split=%d' % k,
                                     save_top_k=2,
                                     monitor='mae_val',
                                     mode='min',
                                     save_weights_only=True)
        lr_monitor = LearningRateMonitor()

        # Trainer
        trainer = Trainer(accelerator='auto',
                          devices=1 if torch.cuda.is_available() else None,
                          max_epochs=72,
                          val_check_interval=3000,
                          callbacks=[model_ckpt, lr_monitor],
                          logger=logger)
        trainer.fit(model, data_module)

        break

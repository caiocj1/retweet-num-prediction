import argparse
import os
import numpy as np
import pandas as pd
import yaml

from models.no_text_mlp import NoTextMLPModel
from models.word2vec_mlp import Word2VecMLPModel
from dataset import RetweetDataModule

import torch.cuda

from pytorch_lightning import Trainer

if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='NoTextMLP')
    parser.add_argument('--weights_path', '-w', required=True)

    args = parser.parse_args()

    # Read config file
    config_path = os.path.join(os.getcwd(), 'config.yaml')
    with open(config_path) as f:
        params = yaml.load(f, Loader=yaml.SafeLoader)

    # Model selection
    model = None
    if args.model == 'NoTextMLP':
        model = NoTextMLPModel()
    elif args.model == 'Word2VecMLP':
        model = Word2VecMLPModel()

    data_module = RetweetDataModule(
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

        prediction = torch.cat(test_results).numpy()
        results.append(prediction)

    final_predictions = np.array(results).mean(0)
    test_results_df = pd.DataFrame(data={'retweets_count': final_predictions})
    test_ids = pd.DataFrame(data_module.test_ids)

    submission = pd.concat([test_ids, test_results_df], axis=1)
    dataset_path = os.getenv('DATASET_PATH')
    submission_path = os.path.join(dataset_path, 'submission.csv')
    submission.to_csv(submission_path, index=False)

# Retweets Prediction

Code made for Kaggle challenge: https://www.kaggle.com/competitions/retweet-prediction-challenge-2022/overview

Part of École Polytechnique's (France) course INF554: Introduction to Machine Learning.

Three models are available: No Text MLP (`'mlp''`), W2V MLP (`'w2v'`, not mentioned in report since it gives worse 
results than the simple No Text MLP) and 
CafayNet (``'conv'``).

All hyperparameters are available and can be changed in the ``config.yaml`` file. Current hyperparameters are the ones
that gave the best result in the leaderboard as explained in the report, for CafayNet.

## Environment creation, tracking training

To create the environment, run ``conda env create -f environment.yaml``.

To track training, ``tensorboard --logdir lightning_logs --bind_all``.

## Launch training

1. Run ``python generate_w2v.py`` to generate word2vec embeddings.
2. ``python run_training.py -v <version_name> -m <model_name>`` (model_name in `['mlp', 'w2v', 'conv']`).
3. If you wish to run a new training with pre-loaded weights, add the option ``-w <path_to_ckpt>``.
4. To generate submission with a trained model, ``python run_prediction.py -m <model_name> -w <path_to_ckpt_folder>``.


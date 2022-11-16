# Retweets Prediction

Code made for Kaggle challenge: https://www.kaggle.com/competitions/retweet-prediction-challenge-2022/overview

Part of École Polytechnique's (France) course INF554: Introduction to Machine Learning.

## Environment creation, tracking training

To create the environment, run ``conda env create -f environment.yaml``.

To track training, ``tensorboard --logdir lightning_logs --bind_all``.

## Launch training

There are two main models: 'NoTextMLP' and 'Word2VecMLP'. The first one removes the text from training data, while the
other takes it into account by concatenating its vector representation to the features.

All hyperparameters are available and can be changed in the ``config.yaml`` file.

Before using the Word2VecMLP model, you must run the ``generate_w2v.py`` script, which will train the embedding model
and save it to the ``models`` folder.
``python generate_w2v.py``

To launch training, ``python run_training.py -v version_name -m model_name``.


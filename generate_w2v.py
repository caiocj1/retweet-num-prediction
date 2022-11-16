import pandas as pd
import gensim, logging
import os
import yaml

if __name__ == '__main__':
    dataset_path = os.getenv('DATASET_PATH')
    train_path = os.path.join(dataset_path, 'train.csv')
    train_df = pd.read_csv(train_path)

    config_path = os.path.join(os.getcwd(), 'config.yaml')
    with open(config_path) as f:
        params = yaml.load(f, Loader=yaml.SafeLoader)
    word2vec_params = params['Word2VecParams']

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    class Corpus(object):
        def __init__(self, text_df: pd.DataFrame):
            self.df = text_df

        def __iter__(self):
            for tweet in self.df['text'].to_list():
                yield tweet.split(' ')

    tweets = Corpus(train_df)
    model = gensim.models.Word2Vec(tweets,
                                   min_count=1,
                                   vector_size=word2vec_params['vector_size'],
                                   workers=4,
                                   epochs=word2vec_params['w2v_epochs'])

    model.save('models/word2vec.model')
    
    print('Saved w2v model')
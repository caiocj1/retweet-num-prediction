import pandas as pd
import gensim, logging
import os

if __name__ == '__main__':
    dataset_path = os.getenv('DATASET_PATH')
    train_path = os.path.join(dataset_path, 'train.csv')
    train_df = pd.read_csv(train_path)

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    class Corpus(object):
        def __init__(self, text_df: pd.DataFrame):
            self.df = text_df

        def __iter__(self):
            for tweet in self.df['text'].to_list():
                yield tweet.split(' ')

    tweets = Corpus(train_df)
    model = gensim.models.Word2Vec(tweets, min_count=1, vector_size=256, workers=4)

    model.save('models/word2vec.model')
    
    print('Saved w2v model')
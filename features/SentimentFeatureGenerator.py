from nltk.sentiment.vader import SentimentIntensityAnalyzer
from features.FeatureGenerator import FeatureGenerator
from nltk import sent_tokenize
import nltk
import numpy as np
import util.FNCData as FNCData


nltk.download('vader_lexicon')


# Takes an average of individual sentiments
def get_sentiment_multiple(paragraph, analyzer):
    sentences = sent_tokenize(paragraph)
    # Average of individual sentiments
    return np.array([get_sentiment(sent, analyzer) for sent in sentences]).mean()


def get_sentiment(sentence, analyzer):
    # Extract the compound score from dict: {neg, neu, pos, compound}
    return analyzer.polarity_scores(sentence)['compound']


# TODO: how to make sure this is same order?
# TODO: Save/load from pickle
class SentimentFeatureGenerator(FeatureGenerator):

    def __init__(self, name='SentimentFeatureGenerator'):
        super(SentimentFeatureGenerator, self).__init__(name)

    # Generates features from a FNC Data object with RAW text
    # The reason we use raw, unfiltered text is because Vader recognizes the importance of punctuation in sentiment
    def generate(self, data, pkl_path):
        assert(isinstance(data, FNCData.FNCData))
        analyzer = SentimentIntensityAnalyzer()

        return (
            [get_sentiment(headline, analyzer) for headline in data.headlines],
            [get_sentiment_multiple(body, analyzer) for body in data.bodies]
        )

    # Load features from pkl
    def load(self, pkl_path):
        pass


if __name__ == '__main__':
    sia = SentimentIntensityAnalyzer()
    sents = ['This is so exciting!!!', 'I hate the heat.', 'Semantic similarity calculated using cosine distance.',
             '----infasdv;asldfsentence. asdf;aksdfl another invalid sents', 'made in 2019; 2020; 2021; invalid']
    joined = ' '.join(sents)
    sentiments = [get_sentiment(sent, sia) for sent in sents]
    print(sentiments)
    print(get_sentiment_multiple(joined, sia))

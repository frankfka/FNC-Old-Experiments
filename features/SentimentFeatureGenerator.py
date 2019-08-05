from nltk.sentiment.vader import SentimentIntensityAnalyzer
from features.FeatureGenerator import FeatureGenerator
import util.FNCData as FNCData


class SentimentFeatureGenerator(FeatureGenerator):

    def __init__(self, name='SentimentFeatureGenerator'):
        super(SentimentFeatureGenerator, self).__init__(name)

    # Generates features from a FNC Data object
    def generate(self, data, pkl_path):
        assert(isinstance(data, FNCData.FNCData))
        pass

    # Load features from pkl
    def load(self, pkl_path):
        pass
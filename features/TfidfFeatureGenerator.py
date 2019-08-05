from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from features.FeatureGenerator import FeatureGenerator
import features.FNCDataGrams as DG


class TfidfFeatureGenerator(FeatureGenerator):

    def __init__(self, name='TfidfFeatureGenerator'):
        super(TfidfFeatureGenerator, self).__init__(name)

    # Generates features from a FNC Grams DF
    def generate(self, data, pkl_path):
        assert(isinstance(data, pd.DataFrame))

        # Create vectorizers - max_df/min_df are cutoffs for document frequency (float -> %, int -> abs # docs)
        # Above max_df -> ex. 0.5 -> ignore words that appear in 50% of the documents or more
        # Below min_df -> ex. 2 -> ignore words that appear than fewer than 2 documents
        v_all = TfidfVectorizer(ngram_range=(1, 3), max_df=0.5, min_df=2)
        # Vectorizer expects a list of strings, so join headline + article for each pair
        all_texts = [' '.join([row[DG.UNI_HEADLINES], row[DG.UNI_BODIES]]) for row in data.iterrows()]
        # Fit all texts vectorizer and create vocabulary
        v_all.fit(all_texts)
        vocab = v_all.vocabulary_

        pass

    # Load features from pkl
    def load(self, pkl_path):
        pass
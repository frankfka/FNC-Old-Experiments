from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from features.FeatureGenerator import FeatureGenerator
import features.FNCDataGrams as DG


class TfidfFeatureGenerator(FeatureGenerator):

    def __init__(self, name='TfidfFeatureGenerator'):
        super(TfidfFeatureGenerator, self).__init__(name)

    # Generates features from a FNC Grams DF -> ([similarities], [headline tfidf]. [body tfidf])
    def generate(self, data, max_df=0.5, min_df=2, pkl_path=None):
        assert (isinstance(data, pd.DataFrame))

        # Create vectorizers - max_df/min_df are cutoffs for document frequency (float -> %, int -> abs # docs)
        # Above max_df -> ex. 0.5 -> ignore words that appear in 50% of the documents or more
        # Below min_df -> ex. 2 -> ignore words that appear than fewer than 2 documents
        v_all = TfidfVectorizer(ngram_range=(1, 3), max_df=max_df, min_df=min_df)
        # Vectorizer expects a list of strings, so join headline + article for each pair
        all_texts = [
            ' '.join(row[DG.UNI_HEADLINES]) + ' ' + ' '.join(row[DG.UNI_BODIES])
            for idx, row in data.iterrows()
        ]
        # Fit all texts vectorizer and create vocabulary
        v_all.fit(raw_documents=all_texts)
        vocab = v_all.vocabulary_

        # Create a vectorizer for headlines based off of complete vocab
        v_head = TfidfVectorizer(ngram_range=(1, 3), max_df=max_df, min_df=min_df, vocabulary=vocab)
        headline_tfidf_list = v_head.fit_transform(
            raw_documents=data[DG.UNI_HEADLINES].map(lambda x: ' '.join(x))  # Re-concats the unigrams into documents
        )  # Creates a TFIDF matrix for headlines
        print(f"Created TFIDF matrix for headlines with shape {headline_tfidf_list.shape}")

        # Do the same for body
        v_body = TfidfVectorizer(ngram_range=(1, 3), max_df=max_df, min_df=min_df, vocabulary=vocab)
        body_tfidf_list = v_body.fit_transform(
            raw_documents=data[DG.UNI_BODIES].map(lambda x: ' '.join(x))
        )
        print(f"Created TFIDF matrix for bodies with shape {body_tfidf_list.shape}")

        # Calculate cosine similarities for each headline/body pair
        cos_similarities = cosine_similarity(headline_tfidf_list, body_tfidf_list)  # n samples x n samples matrix
        cos_similarities = np.diagonal(cos_similarities)

        return cos_similarities, headline_tfidf_list, body_tfidf_list

    # Load features from pkl
    def load(self, pkl_path):
        pass


# TODO: Zip and upload!

if __name__ == '__main__':
    dg = DG.FNCDataGrams(
        txt_headlines=[
            'hello I am frank',
            'this is a sentence with similarity in semantics and such',
            'pickles and oranges don\'t go together but pickles and apples d',
            'stocks are on the rise again',
            'absolute fake news',
            'avocado toast is more expensive than a mortgage'
        ],
        txt_bodies=[
            'this article is about frank',
            'semantic similarity is hard to calculate',
            'fruity punch with pickle and olives are great!! I am so excited to have punch with oranges and grapes',
            'mayweather fights his last championship fight tonight. It will be mayweather versus canelo',
            'faked news on the rise in the united states greatly. Fake news is a big concern',
            'here are the directions to make the perfect avocado toast'
        ]
    )
    fg = TfidfFeatureGenerator()
    sim, h_tfidf, b_tfidf = fg.generate(data=dg.data, max_df=1, min_df=1)
    print(sim)

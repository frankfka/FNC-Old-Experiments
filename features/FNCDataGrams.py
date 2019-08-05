import time
import pandas as pd
from nltk import ngrams

from util.text_processing import tokenize_by_word, clean_tokenized, analyze_pos

UNI_BODIES = 'uni_bodies'
UNI_HEADLINES = 'uni_headlines'
BI_BODIES = 'bi_bodies'
BI_HEADLINES = 'bi_headlines'
TRI_BODIES = 'tri_bodies'
TRI_HEADLINES = 'tri_headlines'


# Tokenize, clean, and lemmatize on a word_tokenized text
def preprocess_and_split(txt):
    tokens = tokenize_by_word(txt)
    tokens = clean_tokenized(tokens)
    tokens = [pair[0] for pair in analyze_pos(tokens, lemmatize=True)]  # Just get the lemmatized word
    return tokens


# Returns a list of n-grams for the given unigrams
def list_of_ngrams(unigrams, n):
    return [gram for gram in ngrams(unigrams, n=n)]


class FNCDataGrams(object):
    """
    Generates uni, bi, and trigrams for FNC data.
    - data: a pandas DF with the constants listed above as column names
    """

    def __init__(self, txt_headlines, txt_bodies, pkl_to=None, pkl_from=None):
        start_time = time.time()

        if pkl_from is not None:
            print("Loading from pickle")
            self.data = pd.read_pickle(pkl_from)
        else:
            print("Creating unigrams")
            uni_bodies = [preprocess_and_split(txt) for txt in txt_bodies]
            uni_headlines = [preprocess_and_split(txt) for txt in txt_headlines]
            print("Creating bigrams")
            bi_bodies = list_of_ngrams(uni_bodies, n=2)
            bi_headlines = list_of_ngrams(uni_headlines, n=2)
            print("Creating trigrams")
            tri_bodies = list_of_ngrams(uni_bodies, n=3)
            tri_headlines = list_of_ngrams(uni_headlines, n=3)
            self.data = pd.DataFrame(data={
                UNI_BODIES: uni_bodies,
                UNI_HEADLINES: uni_headlines,
                BI_BODIES: bi_bodies,
                BI_HEADLINES: bi_headlines,
                TRI_BODIES: tri_bodies,
                TRI_HEADLINES: tri_headlines
            })
            if pkl_to is not None:
                print("Saving grams to pickle")
                try:
                    self.data.to_pickle(pkl_to)
                except Exception as e:
                    print(f"Saving to pickle failed: {e}")

        print(f"FNC Grams Data loaded in {time.time() - start_time}s")

import time
import pandas as pd
from nltk import ngrams

from util.misc import log
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

# TODO: Bigrams, trigrams not actually used anywhere
class FNCDataGrams(object):
    """
    Generates uni, bi, and trigrams for FNC data.
    - data: a pandas DF with the constants listed above as column names
    """

    def __init__(self, txt_headlines, txt_bodies, pkl_to=None, pkl_from=None):
        start_time = time.time()

        if pkl_from is not None:
            log("Loading from pickle")
            self.data = pd.read_pickle(pkl_from)
        else:
            log("Creating unigrams")
            uni_bodies = [preprocess_and_split(txt) for txt in txt_bodies]
            uni_headlines = [preprocess_and_split(txt) for txt in txt_headlines]
            log("Creating bigrams")
            bi_bodies = [list_of_ngrams(txt, n=2) for txt in uni_bodies]  # Each article -> list of (word1, word2)
            bi_headlines = [list_of_ngrams(txt, n=2) for txt in uni_headlines]
            log("Creating trigrams")
            tri_bodies = [list_of_ngrams(txt, n=3) for txt in uni_bodies]  # list of (word1, word2, word3)
            tri_headlines = [list_of_ngrams(txt, n=3) for txt in uni_headlines]
            self.data = pd.DataFrame(data={
                UNI_BODIES: uni_bodies,
                UNI_HEADLINES: uni_headlines,
                BI_BODIES: bi_bodies,
                BI_HEADLINES: bi_headlines,
                TRI_BODIES: tri_bodies,
                TRI_HEADLINES: tri_headlines
            })
            if pkl_to is not None:
                log("Saving grams to pickle")
                try:
                    self.data.to_pickle(pkl_to)
                except Exception as e:
                    log(f"Saving to pickle failed: {e}")

        log(f"FNC Grams Data loaded in {time.time() - start_time}s")


if __name__ == '__main__':
    dg = FNCDataGrams(
        txt_headlines=['hello I am frank', 'this is a sentence', 'pickles and oranges don\'t go together'],
        txt_bodies=['this article is about frank', 'semantic similarity is hard to calculate', 'fruity punch']
    )
    print(dg.data[UNI_HEADLINES])
    print(dg.data[UNI_BODIES])
    print(dg.data[BI_HEADLINES])
    print(dg.data[BI_BODIES])
    print(dg.data[TRI_HEADLINES])
    print(dg.data[TRI_BODIES])

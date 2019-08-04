import gensim
import time
from nltk import word_tokenize

DEFAULT_MAX_SEQ_LEN = 500


class GoogleVectorizer(object):
    """
    Google Vectorization object that allows for text -> vector
    """

    def __init__(self, path='./util/GoogleNews-vectors-negative300.bin.gz'):
        self.path = path
        self.model = None
        self.load()

    def load(self):
        start_time = time.time()
        self.model = gensim.models.KeyedVectors.load_word2vec_format(self.path, unicode_errors='ignore', binary=True)
        print(f"Google word vectors loaded in {time.time() - start_time}s")

    def transform_many(self, list_of_txt, max_seq_len=DEFAULT_MAX_SEQ_LEN):
        return [
            self.transform_one(txt, max_seq_len=max_seq_len) for txt in list_of_txt
        ]

    def transform_one(self, txt, max_seq_len=DEFAULT_MAX_SEQ_LEN):
        # Tokenize text into words, then into vectors
        words = word_tokenize(txt)
        words = words[0:max_seq_len] if len(words) > max_seq_len else words
        return [
            self.model[word] for word in words if word in self.model
        ]


if __name__ == '__main__':
    gv = GoogleVectorizer()

    sent = 'the quick brown fox jumped over the lazy dog'
    t = [sent, 'the. quick, brown! fox,, !']
    transformed = gv.transform_many(t)
    transformed_one = gv.transform_one(sent)
    print(transformed)
    assert(transformed[0] == transformed_one)
    print(len(transformed))
    print(len(transformed[0]))
    print(len(transformed[0][0]))

import time
import pandas as pd

from util.GoogleVectorizer import GoogleVectorizer
from util.misc import log

AGREE_KEY = 'agree'
DISAGREE_KEY = 'disagree'
DISCUSS_KEY = 'discuss'
UNRELATED_KEY = 'unrelated'

stance2idx = {AGREE_KEY: 0, DISAGREE_KEY: 1, DISCUSS_KEY: 2, UNRELATED_KEY: 3}
PICKLE_STANCE = 'stance'
PICKLE_HEADLINE = 'headline'
PICKLE_BODY = 'body'


# Normalize all the counts by cutting out excess data
def balance_stances(df):
    """
    Strips out extra claims so we have a balanced dataset - i.e. # of labels for 0,1,2 are the same
    """
    agree_claims = df[df['Stance'] == AGREE_KEY]
    disagree_claims = df[df['Stance'] == DISAGREE_KEY]
    discuss_claims = df[df['Stance'] == DISCUSS_KEY]
    # unrelated_claims = df[df['Stance'] == UNRELATED_KEY]

    max_index = min([
        len(agree_claims.index),
        len(disagree_claims.index),
        len(discuss_claims.index),
        # len(unrelated_claims.index),
    ])
    log(f"Max index is: {max_index}")
    return pd.concat([
        agree_claims[0: max_index],
        disagree_claims[0: max_index],
        discuss_claims[0: max_index],
        # unrelated_claims[0: max_index]
    ]).sample(frac=1)  # This shuffles


# Get counts (agree, disagree, discuss, unrelated)
def stance_counts(stance_list):
    return (
        len([i for i in stance_list if i == stance2idx[AGREE_KEY]]),
        len([i for i in stance_list if i == stance2idx[DISAGREE_KEY]]),
        len([i for i in stance_list if i == stance2idx[DISCUSS_KEY]])
        # len([i for i in stance_list if i == stance2idx[UNRELATED_KEY]])
    )


# Return (headline, body, stance)
def from_pkl(path, bal_stances=False):
    df = pd.read_pickle(path)
    if bal_stances:
        df = balance_stances(df)
    return df[PICKLE_HEADLINE], df[PICKLE_BODY], df[PICKLE_STANCE]


# Reads a pickle
def to_pkl(headlines, bodies, stances, path):
    try:
        pd.DataFrame(data={
            PICKLE_HEADLINE: headlines,
            PICKLE_BODY: bodies,
            PICKLE_STANCE: stances
        }).to_pickle(path)
    except Exception as e:
        log(f"Saving to pickle failed: {e}")


# Reads the files and returns (headline, body, stance)
def from_files(body_f, stance_f, max_seq_len, vectorizer, pkl_to, bal_stances):
    # Body ID is unique
    bodies_df = pd.read_csv(body_f, index_col='Body ID')
    # Body ID is NOT unique (multiple stances point to same body)
    stances_df = pd.read_csv(stance_f)
    # We don't care about unrelated claims (not relevant to new contest)
    stances_df = stances_df[stances_df['Stance'] != UNRELATED_KEY]
    # Laptop can't process all of these, test on a subset
    # stances_df = stances_df[0:5000]

    # Balance the dataset if required
    if bal_stances:
        stances_df = balance_stances(stances_df)

    headlines = stances_df['Headline']
    bodies = [bodies_df.loc[bodyId, 'articleBody'] for bodyId in stances_df['Body ID']]
    # Convert stances from text to numerical
    stances = [stance2idx[stance] for stance in stances_df['Stance']]
    # Vectorize text if a vectorizer is given
    if vectorizer is not None:
        assert (max_seq_len is not None)
        log("Vectorizing headlines")
        # Convert headlines to shape (# Sequences, SeqLen, Embedding Dim)
        headlines = vectorize_texts(vectorizer, headlines, max_seq_len)
        log("Vectorizing bodies")
        # Get Body from Stance 'Body ID' -> Convert bodies to shape (# Sequences, SeqLen, Embedding Dim)
        bodies = vectorize_texts(vectorizer, bodies, max_seq_len)
    if pkl_to:
        to_pkl(headlines=headlines, bodies=bodies, stances=stances, path=pkl_to)
    return headlines, bodies, stances


# Vectorize a list of texts with given vectorizer v, trimmed to max_seq_len
def vectorize_texts(v, list_of_txt, max_seq_len):
    return v.transform_many(list_of_txt, max_seq_len=max_seq_len)


class FNCData(object):
    """
    Contains Data from the FNC dataset
    - If no vectorizer is passed, then the contained data is the original text with no pre-processing
    - If no vectorizer is passed, max_seq_len is not used
    """

    def __init__(self, max_seq_len=None, vectorizer=None, stance_f=None, body_f=None, bal_stances=False,
                 pkl_to=None, pkl_from=None):
        start_time = time.time()
        if stance_f and body_f:
            self.headlines, self.bodies, self.stances = from_files(max_seq_len=max_seq_len,
                                                                   body_f=body_f,
                                                                   stance_f=stance_f,
                                                                   pkl_to=pkl_to,
                                                                   vectorizer=vectorizer,
                                                                   bal_stances=bal_stances)
        elif pkl_from:
            self.headlines, self.bodies, self.stances = from_pkl(pkl_from, bal_stances=bal_stances)
        else:
            raise ValueError("Incorrect params")
        # self.agree_count, self.disagree_count, self.discuss_count = stance_counts(self.stances)
        log(f"Stance counts: {stance_counts(self.stances)}")
        log(f"FNC Data loaded in {time.time() - start_time}s")

    # Given that this was constructed with pure text properties (i.e. no vectorizer passed). This transforms data to
    # Vector form
    def transform(self, vectorizer, max_seq_len):
        self.headlines = vectorize_texts(vectorizer, self.headlines, max_seq_len)
        self.bodies = vectorize_texts(vectorizer, self.bodies, max_seq_len)


if __name__ == '__main__':
    v = GoogleVectorizer()
    val_news = FNCData(vectorizer=v)
    # Do stuff with val_news

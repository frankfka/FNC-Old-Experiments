import time
import pandas as pd

from util.GoogleVectorizer import GoogleVectorizer

stance2idx = {'agree': 0, 'disagree': 1, 'discuss': 2, 'unrelated': 3}
PICKLE_STANCE = 'stance'
PICKLE_HEADLINE = 'headline'
PICKLE_BODY = 'body'


# Normalize all the counts
def balance_stances(df):
    """
    Strips out extra claims so we have a balanced dataset - i.e. # of labels for 0,1,2 are the same
    """
    agree_claims = df[df['Stance'] == 'agree']
    disagree_claims = df[df['Stance'] == 'disagree']
    discuss_claims = df[df['Stance'] == 'discuss']
    unrelated_claims = df[df['Stance'] == 'unrelated']

    max_index = min([
        len(agree_claims.index),
        len(disagree_claims.index),
        len(discuss_claims.index),
        len(unrelated_claims.index),
    ])
    print(f"Max index is: {max_index}")
    return pd.concat([
        agree_claims[0: max_index],
        disagree_claims[0: max_index],
        discuss_claims[0: max_index],
        unrelated_claims[0: max_index]
    ]).sample(frac=1)  # This shuffles


# Return (headline, body, stance)
def from_pkl(path):
    df = pd.read_pickle(path)
    return df[PICKLE_HEADLINE], df[PICKLE_BODY], df[PICKLE_STANCE]


# Reads a pickle
def to_pkl(headlines, bodies, stances):
    pd.DataFrame(data={
        PICKLE_HEADLINE: headlines,
        PICKLE_BODY: bodies,
        PICKLE_STANCE: stances
    })


# Reads the files and returns (headline, body, stance)
def from_files(body_f, stance_f, max_seq_len, vectorizer, pkl_to):
    # Body ID is unique
    bodies_df = pd.read_csv(body_f, index_col='Body ID')
    # Body ID is NOT unique (multiple stances point to same body)
    stances_df = pd.read_csv(stance_f)
    # Balance the dataset
    print(stances_df['Stance'].value_counts())
    stances_df = balance_stances(stances_df)

    # Convert headlines to shape (# Sequences, SeqLen, Embedding Dim)
    headlines = vectorizer.transform_many(
        stances_df['Headline'],
        max_seq_len=max_seq_len
    )
    # Get Body from Stance 'Body ID' -> Convert bodies to shape (# Sequences, SeqLen, Embedding Dim)
    bodies = vectorizer.transform_many(
        [bodies_df.loc[bodyId, 'articleBody'] for bodyId in stances_df['Body ID']],
        max_seq_len=max_seq_len
    )
    # Convert stances from text to numerical
    stances = [
        stance2idx[stance] for stance in stances_df['Stance']
    ]
    if pkl_to:
        to_pkl(headlines=headlines, bodies=bodies, stances=stances)
    return headlines, bodies, stances


class FNCData(object):

    def __init__(self, max_seq_len=None, vectorizer=None,
                 stance_f=None, body_f=None,
                 pkl_to=None, pkl_from=None):
        start_time = time.time()
        if max_seq_len and vectorizer and stance_f and body_f:
            self.headlines, self.bodies, self.stances = from_files(max_seq_len=max_seq_len,
                                                                   body_f=body_f,
                                                                   stance_f=stance_f,
                                                                   pkl_to=pkl_to,
                                                                   vectorizer=vectorizer)
        elif pkl_from:
            self.headlines, self.bodies, self.stances = from_pkl(pkl_from)
        else:
            raise ValueError("Incorrect params")
        print(f"FNC Data loaded in {time.time() - start_time}s")


if __name__ == '__main__':
    v = GoogleVectorizer()
    val_news = FNCData(vectorizer=v)
    # Do stuff with val_news

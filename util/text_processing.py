import re
import string
import nltk
from nltk.corpus import wordnet as wn
from nltk import WordNetLemmatizer, pos_tag, word_tokenize, sent_tokenize
from nltk.corpus import stopwords

# TODO:
# Don't forget nltk.download('popular')
nltk.download('popular')


# Cleans sentence for processing
def clean_sent(txt):
    """
    Cleans a string for processing - should be used for sentences since punctuation is removed
    - Replace non alphanumeric with whitespace
    - To lowercase
    - Trim whitespace
    """
    return re.sub(r'\W+', ' ', txt).lower().strip()


# Tokenize by Word
def tokenize_by_word(text):
    return word_tokenize(text)


# Tokenize by Setence
def tokenize_by_sentence(text):
    return sent_tokenize(text)


# Cleans a word-tokenized document with given options
def clean_tokenized(
        tokenized,
        remove_stopwords=True,
        remove_punctuation=True,
        lowercase=True
):
    set_to_remove = set()  # Set of strings to remove from tokenized words list

    # Add to the set
    if remove_stopwords:
        stopwords_set = set(stopwords.words('english'))
        set_to_remove = set_to_remove.union(stopwords_set)
    if remove_punctuation:
        set_to_remove = set_to_remove.union(string.punctuation)

    # Process and return
    if lowercase:
        return [w.lower() for w in tokenized if w not in set_to_remove]
    else:
        return [w for w in tokenized if w not in set_to_remove]


# Tags POS's a word-tokenized document
# Has option for lemmatization (ex. flying -> fly)
def analyze_pos(tokenized, lemmatize):
    # This dict converts NLTK POS tags to POS args for Wordnet lemmatizer
    from collections import defaultdict
    pos_to_lemma_arg = defaultdict(lambda: wn.NOUN)  # Default to noun
    pos_to_lemma_arg.update({
        'JJ': wn.ADJ,
        'VB': wn.VERB,
        'RB': wn.ADV
    })

    # Tag words with their part of speech
    tagged_tokens = pos_tag(tokenized)
    # Lemmatize with their POS if needed
    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        tagged_tokens = [
            (lemmatizer.lemmatize(w, pos_to_lemma_arg[pos]), pos)
            for (w, pos) in tagged_tokens
        ]

    # Returns a tuple (word, part_of_speech)
    return tagged_tokens


# Combines a word and its POS (ex. Airplane (Noun) -> Airplane_NN)
def combine_token_pos(tokenized_with_pos):
    return [f"{w}_{pos}" for (w, pos) in tokenized_with_pos]


def tokenize_by_sent(text):
    """Tokenized a given text by sentence -> splits into array of sentences"""
    return nltk.sent_tokenize(text)

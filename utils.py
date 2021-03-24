import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize
from nltk.corpus import wordnet
# nltk.download('punkt')

def append_df_to_JSON(file, df):
    json_df = df.to_json(
        # path_or_buf=file_path,
        orient='records',
        lines=True
    )
    with open(file, 'a', encoding='utf8') as f:
        f.write(json_df)
        f.write("\n")

def tokenize_sentences(df, target_column, new_column=None, drop_target=False):
    if new_column is None:
        new_column = target_column
    print("starting sentence tokenization on", target_column)
    df[new_column] = df[target_column].apply(sent_tokenize)
    if drop_target:
        df.drop(columns=[target_column], inplace=True)
    df.replace("[]", "", inplace=True)

def drop_short_sents(df, column, num_char):
    df.drop(df[df[column].map(len) < num_char].index, inplace=True)

def assign_topic(df, column, vectorizer, topic_model):
    vectorized = vectorizer.transform(df[column])
    topic_scores = topic_model.transform(vectorized)
    df['topic'] = np.argmax(topic_scores, axis=1)

def all_synsets(word, pos=None):
    map = {
        'NOUN': wordnet.NOUN,
        'VERB': wordnet.VERB,
        'ADJ': wordnet.ADJ,
        'ADV': wordnet.ADV
    }
    if pos is None:
        pos_list = [wordnet.VERB, wordnet.ADJ, wordnet.NOUN, wordnet.ADV]
    else:
        pos_list = [map[pos]]
    ret = []
    for pos in pos_list:
        ret.extend(wordnet.synsets(word, pos=pos))
    return ret

def clean_senses(synsets):
    return [x for x in set(synsets) if '_' not in x]

def all_possible_synonyms(word, pos=None):
    ret = []
    for syn in all_synsets(word, pos=pos):
        ret.extend(syn.lemma_names())
    return clean_senses(ret)

def extend_w_synonyms(words, pos=None):
    synonyms = []
    for word in words:
        synonyms.extend(all_possible_synonyms(word, pos))
    return set(words).union(set(synonyms))

def recurse_synonyms(words, pos=None):
    num_words = len(words)
    while True:
        words = extend_w_synonyms(words, pos)
        if num_words == len(words):
            break
        num_words = len(words)
    return words
import os
import time
import pandas as pd
import re

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation

from utils import *

def main():

    start = time.time()

    # load data
    print('loading data...', time.time()-start)
    data_path = "data/glassdoor/rev-anon.parquet"
    df = pd.read_parquet(data_path)
    df.rename(columns={"index": "review_num"}, inplace=True)
    df.info()
    
    # preliminary cleaning
    print('cleaning data...', time.time()-start)
    df.fillna("", inplace=True)
    df.replace(r'\n|\t|\r', ' ', regex=True, inplace=True)
    df.replace(' - ',  ' ', inplace=True)
    df.replace('...',  '. ', inplace=True)
    df.replace('..',  '. ', inplace=True)

    # sentence tokenize
    print('tokenizing sentences...', time.time()-start)
    tokenize_sentences(df, "pros")
    tokenize_sentences(df, "cons")
    tokenize_sentences(df, "feedback")

    # split reviews on sentence
    print('splitting reviews...', time.time()-start)

    df_pros = df[['review_num', 'pros']].copy()
    df_pros['type'] = 1
    df_pros = df_pros.explode('pros')
    df_pros.rename(columns={"pros" : "text"}, inplace=True)

    df_cons = df[['review_num', 'cons']].copy()
    df_cons['type'] = 2
    df_cons = df_cons.explode('cons')
    df_cons.rename(columns={"cons" : "text"}, inplace=True)

    df_feedback = df[['review_num', 'feedback']].copy()
    df_feedback['type'] = 3
    df_feedback = df_feedback.explode('feedback')
    df_feedback.rename(columns={"feedback" : "text"}, inplace=True)

    df = pd.concat([df_pros, df_cons, df_feedback])

    # clean and sort
    df.dropna(axis=0, how='any', inplace=True)
    # drop_short_sents(df, 'text', 3)
    df.sort_index(inplace=True)

    # load topic model
    tfidf_vectorizer = joblib.load("assets/glassdoor/tfidf_vectorizer.jl")
    nmf_fro = joblib.load("assets/glassdoor/nmf_fro.jl")

    # assign topics
    print('assigning glassdoor_topics...', time.time()-start)
    assign_topic(df, 'text', tfidf_vectorizer, nmf_fro)

    # calculate num_words
    print('calculating num_words...', time.time()-start)
    df['num_words'] = df['text'].apply(lambda x: len(x.split()))

    # regex target terms
    print('regexing target terms...', time.time()-start)
    targets = ['account', 'fraud', 'misreport', 'misrepresent']
    targets = extend_w_synonyms(targets)
    regex = '(' + '|'.join(targets) +')'
    df['num_targets'] = df['text'].str.count(regex, re.IGNORECASE)

    # print data info
    df.info()

    # save to stata
    print('saving to stata...', time.time()-start)
    save_path = "data/glassdoor/glassdoor_topics_v2.dta"
    df.to_stata(save_path, version=118, write_index=False)

    print('done', time.time()-start)

if __name__ == '__main__':
    main()
# Preliminary analysis for Toxic Comments competition - Python 3.5.2
# Requires training data in data/train.csv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from pprint import pprint 

df = pd.read_csv("data/train.csv", encoding='utf_8')

df['word_count'] = df['comment_text'].str.split().str.len()

# Show the first ten examples of comments in each category - there will be overlap
# with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth', 10000):
#     print(df[df['toxic'] == 1].head(10))
#     print(df[df['severe_toxic'] == 1].head(10))
#     print(df[df['obscene'] == 1].head(10))
#     print(df[df['threat'] == 1].head(10))
#     print(df[df['insult'] == 1].head(10))
#     print(df[df['identity_hate'] == 1].head(10))

# Show number of comments of different word counts.
word_counts = df['comment_text'].str.split().apply(len).value_counts()
word_counts.sort_index(inplace=True)
with pd.option_context('display.max_rows', None):
    print(word_counts)

# Create columns for all pairs of categories
df['toxic_and_sev_tox'] = df['toxic'] * df['severe_toxic']
df['toxic_and_obscene'] = df['toxic'] * df['obscene']
df['toxic_and_threat'] = df['toxic'] * df['threat']
df['toxic_and_insult'] = df['toxic'] * df['insult']
df['toxic_and_identity_hate'] = df['toxic'] * df['identity_hate']
df['sev_tox_and_obscene'] = df['severe_toxic'] * df['obscene']
df['sev_tox_and_threat'] = df['severe_toxic'] * df['threat']
df['sev_tox_and_insult'] = df['severe_toxic'] * df['insult']
df['sev_tox_and_id_hate'] = df['severe_toxic'] * df['identity_hate']
df['obscene_and_threat'] = df['obscene'] * df['threat']
df['obscene_and_insult'] = df['obscene'] * df['insult']
df['obscene_and_id_hate'] = df['obscene'] * df['identity_hate']
df['threat_and_insult'] = df['threat'] * df['insult']
df['threat_and_id_hate'] = df['threat'] * df['identity_hate']
df['insult_and_id_hate'] = df['insult'] * df['identity_hate']
column_list = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate', 'toxic_and_sev_tox', \
    'toxic_and_obscene', 'toxic_and_threat', 'toxic_and_insult', 'toxic_and_identity_hate', 'sev_tox_and_obscene', \
    'sev_tox_and_threat', 'sev_tox_and_insult', 'sev_tox_and_id_hate', 'obscene_and_threat', 'obscene_and_insult', \
    'obscene_and_id_hate', 'threat_and_insult', 'threat_and_id_hate', 'insult_and_id_hate']

# Create a column for the number of issues each comment has
df['sum_of_issues'] = df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].sum(axis=1)
print("\n\nsum_of_issues")
sum_of_issues = df['sum_of_issues'].value_counts()
sum_of_issues.sort_index(inplace=True)
print(sum_of_issues)

# statistics
print(df.head(10))
print(df.index)
print(df.info())
print(df.describe())

print("\n\nSums for each category:")
print(df[column_list].sum())


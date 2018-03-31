# Converts csv to Google's pretrained 300-dimensional vectors and writes them to tfrecord 
#   train and validation files, so TensorFlow can process the data using a queue.

import datetime
import itertools
import numpy as np
import pandas as pd
import tensorflow as tf
import gensim
from gensim.models.keyedvectors import KeyedVectors

times = []
print("Start ", str(datetime.datetime.now()))
times.append("Start\t\t\t\t" + str(datetime.datetime.now()))

path_to_model = "gnews/GoogleNews-vectors-negative300.bin"

word_vectors = KeyedVectors.load_word2vec_format(path_to_model, binary=True, limit=200000)
max_doc_length = 200
train_percentage = 0.8

# input_df = pd.read_csv("data/train.csv", encoding='utf_8', nrows=3)
input_df = pd.read_csv("data/train.csv", encoding='utf_8')
print("Finished reading csv ", str(datetime.datetime.now()))
times.append("Finished reading csv\t" + str(datetime.datetime.now()))

input_df['separated_text'] = input_df['comment_text'].str.lower().str.replace('[^\w\s]','').str.split()
print("Separated strings ", str(datetime.datetime.now()))
times.append("Separated strings\t" + str(datetime.datetime.now()))

train_df = input_df[0:int(len(input_df.index) * train_percentage)]
validation_df = input_df[int(len(input_df.index) * train_percentage):len(input_df.index)]
validation_df.reset_index(inplace=True)
print("Copied train and validation ", str(datetime.datetime.now()))
times.append("Copied train and validation\t" + str(datetime.datetime.now()))
print("Number of training documents: ", len(train_df.index))
print("Number of validation documents: ", len(validation_df.index))


# These expect to be passed lists of the specified types.
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def write_tfrecords(df, filename):
    writer = tf.python_io.TFRecordWriter(filename)
    count = 0

    for row in df.itertuples():
        input_vector = []

        for word in row.separated_text:
            if word in word_vectors.vocab:
                input_vector.append(word_vectors[word])
            if len(input_vector) == max_doc_length:
                break
        actual_seq_length = len(input_vector)
        flattened_word_vectors = list(itertools.chain(*input_vector))

        feature = {'word_vector': _float_feature(flattened_word_vectors),
                   'sequence_length': _int64_feature([actual_seq_length]),
                   'toxic': _int64_feature([row.toxic]),
                   'severe_toxic': _int64_feature([row.severe_toxic]),
                   'obscene': _int64_feature([row.obscene]),
                   'threat': _int64_feature([row.threat]),
                   'insult': _int64_feature([row.insult]),
                   'identity_hate': _int64_feature([row.identity_hate])}

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())

        count += 1
        if count % 1000 == 0:
            print("Wrote ", count, " records to ", filename)

    writer.close()


print("Writing train records ", str(datetime.datetime.now()))
times.append("Writing train records\t" + str(datetime.datetime.now()))
write_tfrecords(train_df, "tfrecord_train")

print("Writing validation records ", str(datetime.datetime.now()))
times.append("Writing validation records\t" + str(datetime.datetime.now()))
write_tfrecords(validation_df, "tfrecord_validation")

print("\n\nend ", str(datetime.datetime.now()))
print("\n\n")
times.append("end\t\t\t\t" + str(datetime.datetime.now()))

for i in times:
    print(i)



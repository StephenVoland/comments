# This program uses a version of tf-idf to learn a model for solving the Kaggle Toxic Comments challenge.

import csv
from collections import Counter
import copy
import datetime
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pprint import pprint 
from scipy.special import expit
import tensorflow as tf

times = []
print("Start ", str(datetime.datetime.now()))
times.append("Start\t\t\t\t" + str(datetime.datetime.now()))

train_percentage = 0.8
# rows_used = 1050
category_list = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# input_df = pd.read_csv("data/train.csv", encoding='utf_8', nrows=rows_used)
input_df = pd.read_csv("data/train.csv", encoding='utf_8')
rows_used = len(input_df.index) 

input_df['separated_text'] = input_df['comment_text'].str.lower().str.replace('[^\w\s]','').str.split()

train_df = input_df[0:int(len(input_df.index) * train_percentage)].copy()
validation_df = input_df[int(len(input_df.index) * train_percentage):len(input_df.index)].copy()
print("Number of training documents: ", len(train_df.index))
print("Number of validation documents: ", len(validation_df.index))

def update_dict(dict_key, dictionary, update_counter):
    if dict_key not in dictionary:
        dictionary[dict_key] = update_counter
    else:
        dictionary[dict_key].update(update_counter)

# corpus_word_lookup has a key for each word in the corpus.
# The value is a dict stating how many documents contain the word and how many fall into each category.
corpus_word_lookup = dict()

for index, row in train_df.iterrows():
    document_category_values = Counter(dict(row[category_list]))
    single_document = set()
    # Make set of words appearing in document
    for word in row['separated_text']:
        single_document.add(word)
    # for each word in word set, create a Counter and store it in a dict
    document_dictionary = dict()
    for word in single_document:
        temp_counter = copy.deepcopy(document_category_values)
        temp_counter['num_documents'] = 1
        document_dictionary[word] = temp_counter

    for word, values in document_dictionary.items():
        update_dict(word, corpus_word_lookup, values)

# Write train dict to file as CSV
out_df = pd.DataFrame(corpus_word_lookup)
out_df = out_df.T
out_df.to_csv("train_dict.csv")

print("\n\ndict of counts ", str(datetime.datetime.now()))
times.append("dict of counts\t\t\t" + str(datetime.datetime.now()))

train_df['calc_toxic'] = 0.
train_df['calc_severe_toxic'] = 0.
train_df['calc_obscene'] = 0.
train_df['calc_threat'] = 0.
train_df['calc_insult'] = 0.
train_df['calc_identity_hate'] = 0.

# For each document, for each category, calculate a number representing how much that document belongs in that category.
# For example, for each word in the document, divide the number of toxic documents containing the word by the total number of 
#   documents containing the word, then sum the result of this for every word in the document, and divide by the number of words 
#   in the document to get a number representing the average toxicity of words in the document.
# This is based on tf-idf, but is not how it would normally be used.
for index, row in train_df.iterrows():
    document_category_sums = Counter(toxic=0, severe_toxic=0, obscene=0, threat=0, insult=0, identity_hate=0)

    for word in row['separated_text']:
        documents_containing_word = corpus_word_lookup[word]['num_documents']
        document_category_sums.update(Counter({k:v/documents_containing_word for k, v in corpus_word_lookup[word].items()}))

    words_in_document = document_category_sums['num_documents']

    for key, val in document_category_sums.items():
        train_df.loc[index, 'calc_' + key] = val / words_in_document

print("\n\ncalculated values for train ", str(datetime.datetime.now()))
times.append("calculated values for train\t" + str(datetime.datetime.now()))

# Repeat the calculation for the test documents
validation_df['calc_toxic'] = 0.
validation_df['calc_severe_toxic'] = 0.
validation_df['calc_obscene'] = 0.
validation_df['calc_threat'] = 0.
validation_df['calc_insult'] = 0.
validation_df['calc_identity_hate'] = 0.

for index, row in validation_df.iterrows():
    document_category_sums = Counter(toxic=0, severe_toxic=0, obscene=0, threat=0, insult=0, identity_hate=0)

    for word in row['separated_text']:
        if word in corpus_word_lookup:
            documents_containing_word = corpus_word_lookup[word]['num_documents']
            document_category_sums.update(Counter({k:v/documents_containing_word for k, v in corpus_word_lookup[word].items()}))

    words_in_document = document_category_sums['num_documents']

    for key, val in document_category_sums.items():
        if words_in_document == 0:
            validation_df.loc[index, 'calc_' + key] = 0
        else:
            validation_df.loc[index, 'calc_' + key] = val / words_in_document

print("\n\ncalculated values for validation ", str(datetime.datetime.now()))
times.append("calculated values for validation\t" + str(datetime.datetime.now()))

# Calculate multipliers to deal with imbalanced classes.
total_count = int(rows_used * train_percentage)
total_toxic = train_df['toxic'].sum()
total_severe_toxic = train_df['severe_toxic'].sum()
total_obscene = train_df['obscene'].sum()
total_threat = train_df['threat'].sum()
total_insult = train_df['insult'].sum()
total_identity_hate = train_df['identity_hate'].sum()

toxic_multiplier = total_toxic / (total_count - total_toxic)
severe_toxic_multiplier = total_severe_toxic / (total_count - total_severe_toxic)
obscene_multiplier = total_obscene / (total_count - total_obscene)
threat_multiplier = total_threat / (total_count - total_threat)
insult_multiplier = total_insult / (total_count - total_insult)
identity_hate_multiplier = total_identity_hate / (total_count - total_identity_hate)

multiplier_dict = {}
multiplier_dict['toxic_multiplier'] = toxic_multiplier
multiplier_dict['severe_toxic_multiplier'] = severe_toxic_multiplier
multiplier_dict['obscene_multiplier'] = obscene_multiplier
multiplier_dict['threat_multiplier'] = threat_multiplier
multiplier_dict['insult_multiplier'] = insult_multiplier
multiplier_dict['identity_hate_multiplier'] = identity_hate_multiplier

print("total_count: ", total_count)
print("toxic: ", total_toxic)
print("total_severe_toxic: ", total_severe_toxic)
print("total_obscene: ", total_obscene)
print("total_threat: ", total_threat)
print("total_insult: ", total_insult)
print("total_identity_hate: ", total_identity_hate)
print("toxic_multiplier: ", toxic_multiplier)
print("severe_toxic_multiplier: ", severe_toxic_multiplier)
print("obscene_multiplier: ", obscene_multiplier)
print("threat_multiplier: ", threat_multiplier)
print("insult_multiplier: ", insult_multiplier)
print("identity_hate_multiplier: ", identity_hate_multiplier)


def get_batch(data, batch_num, batch_size, x_field, y_field):
    start = batch_num * batch_size
    if (batch_num + 1) * batch_size > len(data.index):
        end = len(data.index)
    else:
        end = (batch_num + 1) * batch_size
    batch_xs = np.expand_dims(data[start:end][x_field], 1)
    batch_ys = np.expand_dims(data[start:end][y_field], 1)
    return batch_xs, batch_ys

def train_all(train_data, validation_data, category_list, multiplier_dict):
    """
    Run training for all categories and return lists of the resulting weight and bias parameters.
    """
    num_examples = len(train_data.index)
    learning_rate = 1
    training_epochs = 100
    batch_size = 100
    display_step = 10
    alpha = 0.0001

    weight_list = []
    bias_list = []

    x = tf.placeholder(tf.float32, [None, 1])
    y = tf.placeholder(tf.float32, [None, 1])
    multiplier = tf.placeholder(tf.float32)

    # Set model weights
    W = tf.Variable(tf.ones([1, 1]))
    b = tf.Variable(tf.zeros([1]))

    pred = tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.matmul(x, W) + b, labels=y)
    # ridge = alpha * (W ** 2 + b ** 2)
    lasso = alpha * (abs(W) + abs(b))
    weighted_pred = tf.nn.weighted_cross_entropy_with_logits(logits=tf.matmul(x, W) + b, targets=y, pos_weight=1/multiplier)
    cost = tf.reduce_mean(weighted_pred) + tf.squeeze(lasso)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for category in category_list:
            sess.run(W.assign([[1]]))
            sess.run(b.assign([0]))

            for epoch in range(training_epochs):
                avg_cost = 0.
                total_batches = math.ceil(num_examples/batch_size)

                for i in range(total_batches):
                    batch_xs, batch_ys = get_batch(train_data, i, batch_size, 'calc_' + category, category)
                    _, c = sess.run([optimizer, cost], 
                            feed_dict={x: batch_xs, y: batch_ys, multiplier: multiplier_dict[category + '_multiplier']})
                    avg_cost += c / total_batches

                    # In case NaNs are generated - should no longer happen.
                    if math.isnan(c):
                        print(batch_xs)
                        print(batch_ys)
                        print(sess.run(weighted_pred, feed_dict={x: batch_xs}))
                        print(sess.run(W))
                        print(sess.run(b))
                        exit()
                if (epoch+1) % display_step == 0:
                    print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
                    print("W:", sess.run(W))
                    print("b:", sess.run(b))

            print(category + " Optimization Finished!")
            print("W:", sess.run(W[0, 0]))
            print("b:", sess.run(b[0]))

            train_batch_xs = np.expand_dims(train_data['calc_' + category], 1)
            train_batch_ys = np.expand_dims(train_data[category], 1)
            validation_batch_xs = np.expand_dims(validation_data['calc_' + category], 1)
            validation_batch_ys = np.expand_dims(validation_data[category], 1)

            correct_predictions = tf.equal(tf.round(pred), y)
            accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
            print("Train Accuracy:", sess.run(accuracy, feed_dict={x: train_batch_xs, y: train_batch_ys}))
            print("Validation Accuracy:", sess.run(accuracy, feed_dict={x: validation_batch_xs, y: validation_batch_ys}))
            weight_list.append(sess.run(W[0, 0]))
            bias_list.append(sess.run(b[0]))

            # tf.reset_default_graph()
            print("finished training " + category + " " + str(datetime.datetime.now()), "\n\n")
            times.append("finished training " + category + "\t\t" + str(datetime.datetime.now()))

    return weight_list, bias_list

weight_list, bias_list = train_all(train_df, validation_df, category_list, multiplier_dict)


# Run on test data and produce results file.
output_df = pd.read_csv("data/test.csv", encoding='utf_8')
# output_df = pd.read_csv("data/test.csv", encoding='utf_8', nrows=rows_used)
output_df['separated_text'] = output_df['comment_text'].str.lower().str.replace('[^\w\s]','').str.split()

# Calculate category information for test data
output_df['calc_toxic'] = 0.
output_df['calc_severe_toxic'] = 0.
output_df['calc_obscene'] = 0.
output_df['calc_threat'] = 0.
output_df['calc_insult'] = 0.
output_df['calc_identity_hate'] = 0.

for index, row in output_df.iterrows():
    document_category_sums = Counter(toxic=0, severe_toxic=0, obscene=0, threat=0, insult=0, identity_hate=0)

    for word in row['separated_text']:
        if word in corpus_word_lookup:
            documents_containing_word = corpus_word_lookup[word]['num_documents']
            document_category_sums.update(Counter({k:v/documents_containing_word for k, v in corpus_word_lookup[word].items()}))

    words_in_document = document_category_sums['num_documents']

    for key, val in document_category_sums.items():
        if words_in_document == 0:
            output_df.loc[index, 'calc_' + key] = 0
        else:
            output_df.loc[index, 'calc_' + key] = val / words_in_document

print("\n\ncalculated values for test ", str(datetime.datetime.now()))
times.append("calculated values for test\t" + str(datetime.datetime.now()))

# Pull out calculated values, multiply them by weight_list, add bias_list, and sigmoid to get predictions
calculation_df = output_df.loc[:, 
                 ['calc_toxic', 'calc_severe_toxic', 'calc_obscene', 'calc_threat', 'calc_insult', 'calc_identity_hate']].copy()
weight_df = pd.Series(weight_list)
bias_df = pd.Series(bias_list)

result_df = expit(calculation_df * weight_df.values + bias_df.values)

result_df.columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
result_df.insert(0, 'id', output_df['id'].copy())

out_filename = "output_" + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + ".csv"
result_df.to_csv(out_filename, index=False)

print("\n\nend ", str(datetime.datetime.now()))
print("\n\n")
times.append("end\t\t\t\t" + str(datetime.datetime.now()))

for i in times:
    print(i)


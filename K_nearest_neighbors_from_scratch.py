import math
import numpy as np
from collections import Counter
import warnings
import pandas as pd
import random


# def euclidean_distance(p: np.array, q: np.array):
#     return math.sqrt(((p[0] - q[0])* (p[0] - q[0])) +
#                      ((p[1] - q[1]) * (p[1] - q[1])))


def k_nearest_neighbors(dataset, predict, k=3):
    if len(dataset) >= k:
        warnings.warn(
            "K is set to a value less than to the total voting groups")
    distances = []
    for group in dataset:
        for features in dataset[group]:
            # Basically the euclidean_distance formula, but numpy like.
            euclidean_distance = np.linalg.norm(
                np.array(features) - np.array(predict))
            distances.append([euclidean_distance, group])
    # sorting distances of the the first to k values.
    votes = [i[1] for i in sorted(distances)[:k]]
    # find the most common occurance.
    # return the group corresponding to it.
    vote_result = Counter(votes).most_common(1)[0][0]
    return vote_result


# Reading in our CSV we got from http://archive.ics.uci.edu/ml/datasets.php
df = pd.read_csv('breast-cancer-wisconsin.data')
# We've got some missing attributes denoted by '?'
# We're replacing it with -99999.
df.replace('?', -99999, inplace=True)
# drop the 'id' column because it has no value for training our classifier.
df.drop(['id'], 1, inplace=True)
# Let's move our data from a dataframe to a list for shuffling purposes.
full_data = df.astype(float).values.tolist()
# random.shuffle occurs inplace
random.shuffle(full_data)

# creating our parameters/variables for our k nearest neighbor algorithm
test_size = 0.2
train_set = {2: [], 4: []}
test_set = {2: [], 4: []}
# taking the first 80% of our shuffled data and putting it into train_data
train_data = full_data[:-int(test_size * len(full_data))]
# taking the last 20% and putting it into test_data
test_data = full_data[-int(test_size * len(full_data)):]

# We're moving our data into the classes that they belong to.
for i in train_data:
    train_set[i[-1]].append(i[:-1])

for i in test_data:
    test_set[i[-1]].append(i[:-1])

correct = 0
total = 0

for group in test_set:
    for data in test_set[group]:
        # throw our train_set, and data into the k_nearest function.
        vote = k_nearest_neighbors(train_set, data, k=3)
        # If it belongs to the right group
        if group == vote:
            correct = correct + 1
        total = total + 1
print('Accuracy: ', correct/total)

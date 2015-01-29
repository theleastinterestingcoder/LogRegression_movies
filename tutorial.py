'''
    tutorial.py 

    An example of text classification. Adapted from chapter 1.3 (Document classification)
    from http://www.nltk.org/book/ch06.html

    Written by: Quan Zhou

'''
import nltk
from nltk.corpus import stopwords
import numpy as np
import json
from random import shuffle
from sklearn import linear_model

def convert_to_vector(review, feature_space, feature_space_index):
    ans = np.zeros(len(feature_space_index))

    for w in review.split():
        if w in feature_space:
            idx = feature_space_index[w]
            ans[idx] = 1

    return ans

def interpret_label(label):
    if label == 'fresh':
        return 1
    else:
        return 0
# ----------------------------
# Main starts here
# -----------------

# Load the file
fname = 'items.json'
raw_data = json.load(open(fname, 'r'))

# Collect a histogram of all the words
fdist = nltk.FreqDist()


# (1) Determine what our feature space will look like
for page in raw_data: 
    for review in page['reviews']:
        tmp = nltk.FreqDist(review[1].split())  
        fdist.update(tmp)

# We want an informative feature space of the 2000 most common words. 
feature_space = [tuplet[0] for tuplet in fdist.most_common(2000)]
feature_space = [word.lower() for word in feature_space]
# Filter out the most common (and probably uninformative words)
feature_space = [word for word in feature_space if tuplet[0] not in stopwords.words('english')]
feature_space = set(feature_space)

# Create a dictionary to make the search go faster
feature_space_index = {}
for (i,word) in enumerate(feature_space):
    feature_space_index[word] = i

# (2) Convert each review into a datapoint
dataset = []
for (i,page) in enumerate(raw_data): 
    for review in page['reviews']:
        point = convert_to_vector(review[1], feature_space, feature_space_index)
        label = interpret_label(review[0])
        dataset.append((point, label))

shuffle(dataset)

# Divide dataset into testset and a trainset
trainset = dataset[0:len(dataset)*3/4]
testset  = dataset[len(dataset)*3/4:]

# (3) Train a logistic regression classifier on the test set
x_train = [i[0] for i in trainset]
y_train = [i[1] for i in trainset]
x_test = [i[0] for i in testset]
y_test = [i[1] for i in testset]


reg_guess = 1e-1
log_class = linear_model.LogisticRegression(C=reg_guess, penalty='l2', tol=1e-6)
log_class.fit(x_train, y_train)


# (4) Now that we have our classifier, let's get its accuracy
correct = 0
for (x,y) in zip(x_test, y_test):
    p = log_class.predict(x)
    if p == y:
        correct = correct + 1

print "Accuracy is %s" % (float(correct)/len(x_test))

# (5) My prediction
review = 'This movie was terrible'
point = convert_to_vector(review, feature_space, feature_space_index)
p = log_class.predict(point)[0]

if p  == 1:
    print "Fresh!"
else:
    print "Rotten!"


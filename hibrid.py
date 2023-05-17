import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.semi_supervised import LabelPropagation



# REVIEWS !!
print("\nREVIEWS\n")
# Load the CSV file into a pandas DataFrame
data = pd.read_csv('data/reviews_mixed.csv')

# Extract the text column from the DataFrame
text_data = data['Text'].tolist()

inputs = text_data
# labels = []
# for entry in data['Sentiment']:
#     if entry == 'positive':
#         labels.append(np.array([0, 1]))
#     else:
#         labels.append(np.array([1, 0]))
# labels = np.array(labels)
labels = data['Sentiment'].tolist()

indexes = [i for i in range(len(inputs))]
trainSample = np.random.choice(indexes, int(0.8 * len(inputs)), replace=False)
validationSample = [i for i in indexes if not i in trainSample]
trainInputs = [inputs[i] for i in trainSample]
trainLabels = [labels[i] for i in trainSample]
validationInputs = [inputs[i] for i in validationSample]
validationLabels = [labels[i] for i in validationSample]

vectorizer = TfidfVectorizer(ngram_range=(1,3), max_features=100)

trainFeatures = vectorizer.fit_transform(trainInputs)
testFeatures = vectorizer.transform(validationInputs)
y = []
for output in trainLabels:
    if output == 'positive':
        y.append(0)
    else:
        y.append(1)

model = LabelPropagation(kernel='knn', n_neighbors=10, max_iter=100)
r = np.random.RandomState(int(time.time()))
random_unlabeled_points = r.rand(len(y)) < 0.3
labels = np.copy(y)
labels[random_unlabeled_points] = -1
model.fit(trainFeatures.toarray(), labels)

predicted = model.predict(testFeatures)
v = []
for l in validationLabels:
    if l == 'positive':
        v.append(0)
    else:
        v.append(1)
validationLabels = v

print("Accuracy :", accuracy_score(validationLabels, predicted))


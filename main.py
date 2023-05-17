import time

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.semi_supervised import LabelPropagation

from kMeans.myKMeans import MyKMeans

# REVIEWS !!
print("\nREVIEWS\n")
# Load the CSV file into a pandas DataFrame
data = pd.read_csv('data/reviews_mixed.csv')

# Extract the text column from the DataFrame
text_data = data['Text'].tolist()

# Convert the text data into numerical feature vectors
vectorizer = TfidfVectorizer()
features = vectorizer.fit_transform(text_data)
features_array = features.toarray()

inputs = features_array
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

t = []
for img in trainInputs:
    t.append(img)
trainInputs = np.array(t)
t = []
for img in validationInputs:
    t.append(np.array(img))
validationInputs = np.array(t)

# Define the number of clusters
num_clusters = 2

kmeans = MyKMeans()
kmeans.fit(trainInputs, 2)

predicted = kmeans.predict(validationInputs)
# for i in range(0, len(predicted)):
#     print(data['Text'][i], predicted[i])

acc1 = 0
for i in range(0, len(predicted)):
    if validationLabels[i] == 'positive' and predicted[i] == 0:
        acc1+=1
    if validationLabels[i] == 'negative' and predicted[i] == 1:
        acc1+=1
print("Accuracy (0-positiv 1-negativ): ", acc1/len(predicted))

acc2 = 0
for i in range(0, len(predicted)):
    if validationLabels[i] == 'positive' and predicted[i] == 1:
        acc2+=1
    if validationLabels[i] == 'negative' and predicted[i] == 0:
        acc2+=1
print("Accuracy (1-positiv 0-negativ): ", acc2/len(predicted))

# IRIS !! [0,1,2] <=> [setosa, versicolor, virginica]
print("\nIRIS\n")
data = load_iris()
outputs = []
for op in data['target']:
    v = [0 for i in range(3)]
    v[op] = 1
    outputs.append(np.array(v))

labels = np.array(outputs) # one hot
inputs = data['data']


# impartire test / train
indexes = [i for i in range(len(inputs))]
trainSample = np.random.choice(indexes, int(0.8 * len(inputs)), replace=False)
validationSample = [i for i in indexes if not i in trainSample]
trainInputs = [inputs[i] for i in trainSample]
trainLabels = [labels[i] for i in trainSample]
validationInputs = [inputs[i] for i in validationSample]
validationLabels = [labels[i] for i in validationSample]

t = []
for img in trainInputs:
    t.append(img)
trainInputs = np.array(t)
t = []
for img in validationInputs:
    t.append(np.array(img))
validationInputs = np.array(t)

trainOutputs = []
for l in trainLabels:
    trainOutputs.append(np.argmax(l))



kmeans = MyKMeans()
kmeans.fit(trainInputs, 3)
predicted = kmeans.predict(validationInputs)
for i in range(0, len(predicted)):
    print(validationLabels[i], predicted[i])

mappings = [{0 : 0, 1 : 1, 2 : 2}, {0 : 1, 1 : 0, 2 : 2}, {0 : 2, 1 : 0, 2 : 1}, {0 : 0, 1 : 2, 2 : 2}, {0 : 1, 1 : 2, 2: 0}, {0 : 2, 1 : 1, 2 : 0}]
for mapping in mappings:
    acc = 0
    for i in range(len(predicted)):
        if validationLabels[i][mapping[int(predicted[i])]] == 1:
            acc+=1
    print("Acuratete IRIS Mapping (" + str(mapping) + ") : " + str(acc/len(predicted)))


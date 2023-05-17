# REVIEWS !!
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from regression.batch_logistic_regression import MyBatchLogisticRegression

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
labels = []
for entry in data['Sentiment']:
    if entry == 'positive':
        labels.append(0)
    else:
        labels.append(1)
labels = np.array(labels)


indexes = [i for i in range(len(inputs))]
trainSample = np.random.choice(indexes, int(0.8 * len(inputs)), replace=False)
validationSample = [i for i in indexes if not i in trainSample]
trainInputs = [inputs[i] for i in trainSample]
trainLabels = [labels[i] for i in trainSample]
validationInputs = [inputs[i] for i in validationSample]
validationLabels = [labels[i] for i in validationSample]

print(trainInputs)

regressor = MyBatchLogisticRegression()
weights, intercept = regressor.fit(trainInputs, trainLabels)

predicted = regressor.predict(validationInputs)
print(predicted)

acc1 = 0
for i in range(0, len(predicted)):
    if validationLabels[i] == predicted[i]:
        acc1+=1
print("Accuracy (0-positiv 1-negativ): ", acc1/len(predicted))

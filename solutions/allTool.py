import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Load the CSV file into a pandas DataFrame
data = pd.read_csv('../data/reviews_mixed.csv')

# Extract the text column from the DataFrame
text_data = data['Text'].tolist()

# Convert the text data into numerical feature vectors
vectorizer = TfidfVectorizer()
features = vectorizer.fit_transform(text_data)

# Define the number of clusters
num_clusters = 2

# Run k-means clustering
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(features)

# Get the cluster labels
labels = kmeans.labels_

# Add the cluster labels to the DataFrame
data['cluster'] = labels

# Print the results
accuracy = 0
for i in range(0, len(data['Text'])):
    print(data['Text'][i], data['cluster'][i])
    if data['Sentiment'][i] == "negative" and data['cluster'][i] == 0:
        accuracy+=1
    if data['Sentiment'][i] == "positive" and data['cluster'][i] == 1:
        accuracy += 1

print("ACCURACY : ", accuracy / len(data['Text'])) # best 70%

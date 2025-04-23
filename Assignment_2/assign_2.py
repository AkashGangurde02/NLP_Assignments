import os
import nltk
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import heapq
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec

# Ensure required nltk resources are available
nltk.download('punkt')

# Function to read the file safely
def read_file(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found in {os.getcwd()}")
        exit()
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Provide the correct path to the file
file_path = "assign_1.txt"  # Change this if needed
text = read_file(file_path)

# Tokenize sentences
dataset = nltk.sent_tokenize(text)

# Preprocess dataset (convert to lowercase, remove special characters)
for i in range(len(dataset)):
    dataset[i] = dataset[i].lower()
    dataset[i] = re.sub(r'\W', ' ', dataset[i])
    dataset[i] = re.sub(r'\s+', ' ', dataset[i])

# Bag-of-Words Model (Count Occurrence)
word2count = Counter()
for data in dataset:
    words = nltk.word_tokenize(data)
    word2count.update(words)

# Select the top 100 frequent words
freq_words = heapq.nlargest(100, word2count, key=word2count.get)

# Create Bag-of-Words Matrix (Binary Representation)
X = []
for data in dataset:
    vector = [1 if word in nltk.word_tokenize(data) else 0 for word in freq_words]
    X.append(vector)
X = np.array(X)

# Visualizing the BoW matrix
plt.figure(figsize=(10, 12))
sns.heatmap(X, cmap="coolwarm")
plt.title("Bag of Words Representation")
plt.show()

# TF-IDF Implementation
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(dataset).toarray()

# Display the top TF-IDF words
feature_names = vectorizer.get_feature_names_out()
tfidf_scores = np.sum(X_tfidf, axis=0)
sorted_indices = np.argsort(tfidf_scores)[::-1][:10]

print("Top 10 Words by TF-IDF Score:")
for i in sorted_indices:
    print(f"{feature_names[i]}: {tfidf_scores[i]:.4f}")

# Word2Vec Embeddings
tokenized_sentences = [nltk.word_tokenize(sent) for sent in dataset]
word2vec_model = Word2Vec(sentences=tokenized_sentences, vector_size=100, window=5, min_count=1, workers=4)

# Example: Get Word Embedding for a Specific Word
example_word = "data"  # Change to any word in your text
if example_word in word2vec_model.wv:
    print(f"Word2Vec Embedding for '{example_word}':\n", word2vec_model.wv[example_word])
else:
    print(f"Word '{example_word}' not found in Word2Vec model.")

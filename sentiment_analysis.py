
# sentiment_analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import nltk
from nltk.corpus import stopwords


stop_words = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",
    "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his',
    'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself',
    'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom',
    'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be',
    'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a',
    'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at',
    'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during',
    'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on',
    'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
    'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
    'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
    'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now',
    'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't",
    'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven',
    "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn',
    "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren',
    "weren't", 'won', "won't", 'wouldn', "wouldn't"
}

# Load the dataset
df = pd.read_csv("data/IMDB_Dataset.csv") 


# Display basic info
print("Dataset Loaded Successfully!")
print(df.head())
print(df['sentiment'].value_counts())

# Visualize sentiment distribution
df['sentiment'].value_counts().plot(kind='bar', color=['skyblue', 'orange'])
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show(block=True)


# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char.isalpha() or char == ' '])  #adding only alphabets and spaces 
    text = ' '.join([word for word in text.split() if word not in stop_words])  #removing stop words
    return text

# Apply cleaning
df['review'] = df['review'].apply(clean_text) #adding clean text to the table

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    df['review'], df['sentiment'], test_size=0.2, random_state=42
)

# TF-IDF vectorization  term frequency inverse document frequency
vectorizer = TfidfVectorizer(max_features=5000)  # use only top 5000 words
X_train_tfidf = vectorizer.fit_transform(X_train) #splits and transforms to numbers
X_test_tfidf = vectorizer.transform(X_test)     # if any new words appears from traiining it doesnt learns it skips


# Model training to make the predictions
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)


# Predictions
y_pred = model.predict(X_test_tfidf)


# Evaluation
print("\n Model Evaluation Results:")
print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


#checking output
# ---- Predict Sentiment for New Reviews ----

def predict_sentiment(new_review):
    # Clean the text just like training data
    cleaned = clean_text(new_review)

    # Convert to TF-IDF features (using same vectorizer)
    vectorized = vectorizer.transform([cleaned])

    # Predict sentiment
    prediction = model.predict(vectorized)[0]

    print(f"\nReview: {new_review}")
    print(f"Predicted Sentiment: {prediction}")

# Try it out
predict_sentiment("The movie was really amazing and emotional!")
predict_sentiment("This was a boring movie with terrible acting.")
predict_sentiment("The movie was really amazing")

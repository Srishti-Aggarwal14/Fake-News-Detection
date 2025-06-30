import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

nltk.download('stopwords')

# -----------------------------------
# Bigger toy dataset (20 samples)
# -----------------------------------
data = {
    'text': [
        "Government confirms COVID-19 vaccines are safe and tested.",
        "Fake news claims aliens have invaded New York.",
        "NASA launches new Mars rover successfully.",
        "Hoax: chocolate cures COVID-19 overnight!",
        "Local man wins lottery twice in one week — true story.",
        "Clickbait: you won’t believe this celebrity secret!",
        "New policy to support small businesses approved.",
        "Fake reports say drinking bleach kills virus.",
        "Scientists discover new species in Amazon rainforest.",
        "False rumor: school closing due to ghost sightings.",
        "WHO says new variant is under control.",
        "Misleading article: moon landing was fake.",
        "Election results confirmed by authorities.",
        "False claim: 5G towers spread coronavirus.",
        "Economic recovery faster than expected.",
        "Hoax: eating garlic prevents COVID-19.",
        "New vaccine reduces infection by 90%.",
        "Fake headline: zombies spotted in Paris.",
        "Electric cars sales reach new record.",
        "Conspiracy theory: earth is flat proven."
    ],
    'label': [
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1
    ]
}

df = pd.DataFrame(data)
print(df.head())

# -----------------------------------
# NLP cleaning
# -----------------------------------
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = ' '.join([stemmer.stem(word) for word in text.split() if word not in stop_words])
    return text

df['text'] = df['text'].apply(clean_text)

# -----------------------------------
# Vectorize
# -----------------------------------
vectorizer = TfidfVectorizer(max_features=500)
X = vectorizer.fit_transform(df['text'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------------
# Model comparison
# -----------------------------------
models = {
    "Logistic Regression": LogisticRegression(),
    "Naive Bayes": MultinomialNB(),
    "SVM": LinearSVC()
}

for name, model in models.items():
    print(f"\n----- {name} -----")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Confusion Matrix Heatmap
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# -----------------------------------
# Word Cloud
# -----------------------------------
text_data = ' '.join(df['text'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_data)

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud of News Text")
plt.show()

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from french_lefff_lemmatizer.french_lefff_lemmatizer import FrenchLefffLemmatizer

# Download required NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Initialize French NLP tools
stemmer = SnowballStemmer('french')
lemmatizer = FrenchLefffLemmatizer()
stop_words = set(stopwords.words('french'))

# Add culinary-specific stop words if needed
extra_stopwords = {'aux', 'à', 'la', 'le', 'de', 'du', 'des', 'avec'}
stop_words.update(extra_stopwords)

# Load sentence transformer model
model = SentenceTransformer('dangvantuan/sentence-camembert-base')

# Sample DataFrame
data = {'plat': ["ratatouille + tajine aux artichauts + couscous",
                 "poulet rôti + saumon grillé + gratin dauphinois",
                 "salade niçoise + soupe à l'oignon",
                 "steak frites + bourguignon + confit de canard",
                 "tarte aux poireaux + quiche lorraine + flamiche"]}
df = pd.DataFrame(data)

def preprocess(text):
    # Tokenize and clean text
    tokens = nltk.word_tokenize(text.lower(), language='french')
    processed = []
    for token in tokens:
        if token.isalpha() and token not in stop_words:
            # Use either lemmatization or stemming
            lemma = lemmatizer.lemmatize(token)
            processed.append(stemmer.stem(lemma))  # Combine both approaches
    return ' '.join(processed)

# Extract and preprocess all dishes
all_dishes = []
for row in df['plat']:
    dishes = [d.strip() for d in row.split('+')]
    all_dishes.extend([preprocess(dish) for dish in dishes])

unique_dishes = list(set(all_dishes))

# Generate embeddings and cluster
dish_embeddings = model.encode(unique_dishes)
num_clusters = 5  # Adjust based on your data
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(dish_embeddings)

# Get cluster names using TF-IDF
def get_cluster_names(dishes, labels, n_terms=3):
    from sklearn.feature_extraction.text import TfidfVectorizer
    cluster_names = {}
    
    for cluster_id in set(labels):
        cluster_dishes = [dishes[i] for i, lbl in enumerate(labels) if lbl == cluster_id]
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(cluster_dishes)
        features = vectorizer.get_feature_names_out()
        scores = np.asarray(X.mean(axis=0)).ravel()
        top_terms = [features[i] for i in np.argsort(scores)[-n_terms:][::-1]]
        cluster_names[cluster_id] = ' '.join(top_terms)
    
    return cluster_names

category_mapping = get_cluster_names(unique_dishes, kmeans.labels_)

# Categorization function
def categorize_dish(dish):
    clean_dish = preprocess(dish)
    emb = model.encode([clean_dish])
    return category_mapping[kmeans.predict(emb)[0]]

# Process the dataframe
def process_row(row):
    dishes = [d.strip() for d in row.split('+')]
    return ' + '.join([f"{d} ({categorize_dish(d)})" for d in dishes])

df['categorized_plat'] = df['plat'].apply(process_row)
print(df[['plat', 'categorized_plat']])

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sentence_transformers import SentenceTransformer
import spacy

# Load French NLP model
nlp = spacy.load("fr_core_news_sm")

# Load French sentence embedding model
model = SentenceTransformer('dangvantuan/sentence-camembert-base')

# Sample DataFrame
data = {'plat': ["ratatouille + tajine aux artichauts + couscous",
                 "poulet rôti + saumon grillé + gratin dauphinois",
                 "salade niçoise + soupe à l'oignon",
                 "steak frites + bourguignon + confit de canard",
                 "tarte aux poireaux + quiche lorraine + flamiche"]}
df = pd.DataFrame(data)

# Preprocessing function
def preprocess(text):
    doc = nlp(text.lower())
    return " ".join([token.lemma_ for token in doc if not token.is_stop and token.is_alpha])

# Extract and preprocess all dishes
all_dishes = []
for row in df['plat']:
    dishes = [d.strip() for d in row.split('+')]
    all_dishes.extend([preprocess(dish) for dish in dishes])

unique_dishes = list(set(all_dishes))

# Generate embeddings
dish_embeddings = model.encode(unique_dishes)

# Determine optimal clusters (simplified version)
num_clusters = 5  # Adjust based on your data

# Cluster dishes
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(dish_embeddings)
cluster_labels = kmeans.labels_

# Create cluster to category mapping
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

category_mapping = get_cluster_names(unique_dishes, cluster_labels)

# Function to categorize a new dish
def categorize_dish(dish, preprocessed=False):
    if not preprocessed:
        dish = preprocess(dish)
    emb = model.encode([dish])
    cluster = kmeans.predict(emb)[0]
    return category_mapping[cluster]

# Process original DataFrame
def process_row(row):
    dishes = [d.strip() for d in row.split('+')]
    categorized = []
    for dish in dishes:
        original_dish = dish
        clean_dish = preprocess(dish)
        category = categorize_dish(clean_dish, preprocessed=True)
        categorized.append(f"{original_dish} ({category})")
    return ' + '.join(categorized)

df['categorized_plat'] = df['plat'].apply(process_row)
print(df[['plat', 'categorized_plat']])

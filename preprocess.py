import pandas as pd
import re
try:
    import nltk
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "nltk"])
    import nltk
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Download NLTK resources (only required once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)
try:
    nltk.data.find('stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Load dataset
df = pd.read_csv('projects_dataset.csv')

# Define required columns (Updated: Removed 'referencepaper_link')
required_columns = ['projects_id', 'project_title', 'project_description', 'technology_stack', 'domain', 'skills_required',
                    'difficulty_level', 'prerequisties', 'project_type', 'duration', 'tags', 'reference_link',
                    'resource_link', 'user_ratings_avg']

# Check for missing columns
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    logging.warning(f"⚠️ Warning: Missing columns in dataset: {', '.join(missing_columns)}")
    for col in missing_columns:
        df[col] = ''  # Fill missing columns with empty values
else:
    logging.info("✅ All required columns are present.")

# Convert all column names to lowercase
df.columns = df.columns.str.lower()

# Convert relevant columns to lowercase
for col in ['tags', 'skills_required', 'domain']:
    if col in df.columns:
        df[col] = df[col].astype(str).str.lower()

# Preprocessing function for text
def preprocess_text(text):
    if pd.isna(text) or not isinstance(text, str):  # Handle NaN values
        return ''
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalnum() and token not in stop_words]
    return ' '.join(tokens)

# Apply preprocessing
for col in ['project_description', 'technology_stack', 'skills_required', 'tags']:
    if col in df.columns:
        df[col] = df[col].apply(preprocess_text)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english')

# Handle missing descriptions before TF-IDF
df['project_description'] = df['project_description'].fillna('')
descriptions = df['project_description'].tolist()
tfidf_matrix = vectorizer.fit_transform(descriptions)

# Create a lookup dictionary for project descriptions
tfidf_matrix_lookup = {i: desc for i, desc in enumerate(df['project_description'])}

# Save processed data
try:
    df.to_pickle('projects_dataframe.pkl')
    with open('tfidf_matrix.pkl', 'wb') as f:
        pickle.dump(tfidf_matrix, f)
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    with open('tfidf_matrix_lookup.pkl', 'wb') as f:
        pickle.dump(tfidf_matrix_lookup, f)

    logging.info("✅ Preprocessing and saving complete.")
    print("Preprocessing complete. Files saved: projects_dataframe.pkl, tfidf_matrix.pkl, vectorizer.pkl, tfidf_matrix_lookup.pkl.")

except Exception as e:
    logging.error(f"❌ Error during saving: {e}")
    print(f"❌ Error during saving: {e}")
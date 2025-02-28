import streamlit as st
import pandas as pd
import re
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize text processing tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Load Data (Handle Missing Files)
try:
    with open('projects_dataframe.pkl', 'rb') as f:
        df = pickle.load(f)
    with open('tfidf_matrix.pkl', 'rb') as f:
        tfidf_matrix = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('tfidf_matrix_lookup.pkl', 'rb') as f:
        tfidf_matrix_lookup = pickle.load(f)
except FileNotFoundError:
    st.error("❌ Pickle files not found. Please run `preprocess.py` first.")
    st.stop()

# Search Function
def search_across_all_columns(input_value, df):
    input_value = input_value.lower()
    input_value = re.split(r'[,\s]+', input_value)  # Split by comma or space
    mask = pd.Series([False] * len(df))
    columns_to_search = ['tags', 'skills_required', 'domain', 'project_description', 'technology_stack']
    for column in columns_to_search:
        for term in input_value:
            mask |= df[column].str.contains(term, case=False, na=False)
    return df[mask]

# Recommendation Function
def recommend_similar_projects(project_description, df, tfidf_matrix, vectorizer, tfidf_matrix_lookup):
    try:
        project_index = list(tfidf_matrix_lookup.values()).index(project_description)
    except ValueError:
        st.write("No matching project found in dataset.")
        return []
    cosine_similarities = cosine_similarity(tfidf_matrix[project_index], tfidf_matrix).flatten()
    similar_indices = cosine_similarities.argsort()[-6:-1][::-1]  # Get top 5 (excluding the project itself)
    return df.iloc[similar_indices]['project_title'].tolist()

# Streamlit App Title
st.title("🔍 Project Recommendation System")

# Search Bar
search_term = st.text_input("Enter search terms:", key='search_bar')

# Search Functionality
if search_term:
    filtered_projects = search_across_all_columns(search_term, df)
    if filtered_projects.empty:
        st.write("No projects found with the specified criteria.")
    else:
        st.write(f"Projects matching '{search_term}':")
        for project_title in filtered_projects['project_title']:
            with st.expander(project_title):
                project_details = df[df['project_title'] == project_title].iloc[0]
                st.write(f"**Description:** {project_details['project_description']}")
                st.write(f"**Technology Stack:** {project_details['technology_stack']}")
                st.write(f"**Domain:** {project_details['domain']}")
                st.write(f"**Skills Required:** {project_details['skills_required']}")
                st.write(f"**Difficulty Level:** {project_details['difficulty_level']}")
                st.write(f"**Duration:** {project_details['duration']}")
                st.write(f"**Prerequisites:** {project_details['prerequisties']}")
                st.write(f"**Project Type:** {project_details['project_type']}")
                if pd.notna(project_details['reference_link']) and project_details['reference_link'].strip():
                    st.write(f"[Reference Link]({project_details['reference_link']})")
                if pd.notna(project_details['resource_link']) and project_details['resource_link'].strip():
                    st.write(f"[Resource Link]({project_details['resource_link']})")
                st.write(f"**User Ratings Average:** {project_details['user_ratings_avg']}")
                
                project_description = project_details['project_description']
                similar_projects = recommend_similar_projects(project_description, df, tfidf_matrix, vectorizer, tfidf_matrix_lookup)
                if similar_projects:
                    st.write("### Recommended Similar Projects:")
                    for i, similar_project in enumerate(similar_projects, 1):
                        st.write(f"{i}. {similar_project}")
                else:
                    st.write("No similar projects found.")

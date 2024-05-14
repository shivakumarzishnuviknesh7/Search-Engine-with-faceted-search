import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Read CSV file with specified encoding
file_path = r'C:\Users\Dell\Desktop\Msc applied computer science\Semester 2\HMI (Project)\OM_Project\course_module.csv'
df = pd.read_csv(file_path, encoding='latin1')

# Preprocess data
# Remove NaN values in the 'Module Name' column
df = df.dropna(subset=['Module Name'])

# Vectorize text data using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['Module Name'].astype(str))  # Convert to string to handle NaN values

# Define function to calculate cosine similarity
def calculate_similarity(query, data):
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, data)
    return similarities

# Define Streamlit app
st.title('Course Search')

# User input for search query
query = st.text_input('Enter course name:')

# Perform search using cosine similarity and display top 3 most similar module names
if st.button('Search (Cosine Similarity)'):
    similarities = calculate_similarity(query, X)
    # Get indices of top 3 most similar module names
    top_3_indices = similarities.argsort(axis=1).flatten()[-3:][::-1]
    st.write('Top 3 most similar module names (Cosine Similarity):')
    for idx in top_3_indices:
        module_name = df.iloc[idx]['Module Name']
        similarity_score = similarities[0, idx]
        st.write(module_name, '-', similarity_score)

# Faceted search options
facets = ["Teaching Methods", "Semester", "Type of Course"]

# Sidebar for facet selection
selected_facets = {}
for facet in facets:
    selected_facets[facet] = st.sidebar.multiselect(facet, df[facet].unique())

# Filter data based on selected facets
filtered_df = df.copy()
for facet, values in selected_facets.items():
    if values:
        filtered_df = filtered_df[filtered_df[facet].isin(values)]

# Display filtered results
if st.sidebar.button('Search (Faceted Search)'):
    st.title('Faceted Search Results')
    st.write(filtered_df)

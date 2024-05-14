"""

##don not touch

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

# Search function to find partial matches
def search_module(query, module_names):
    results = []
    for module_name in module_names:
        if query.lower() in module_name.lower():
            results.append(module_name)
    return results

# Perform search and display results
if st.button('Search'):
    results = search_module(query, df['Module Name'])
    if results:
        st.write('Search Results:')
        for result in results:
            st.write(result)
    else:
        st.write('No matching modules found.')


"""

######################################################################################################################################################################


"""

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

# Perform search and display top 3 most similar module names
if st.button('Search'):
    similarities = calculate_similarity(query, X)
    # Get indices of top 3 most similar module names
    top_3_indices = similarities.argsort(axis=1).flatten()[-3:][::-1]
    st.write('Top 3 most similar module names:')
    for idx in top_3_indices:
        module_name = df.iloc[idx]['Module Name']
        similarity_score = similarities[0, idx]
        st.write(module_name, '-', similarity_score)


"""











import streamlit as st
import concurrent.futures
from functools import partial
import numpy as np
from io import StringIO
import sys
import time

# File Imports
from embedding import get_embeddings  # Ensure this file/module is available
from preprocess import filtering  # Ensure this file/module is available
from search import *

# Cosine Similarity Function
def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    dot_product = np.dot(vec1, vec2)
    magnitude_vec1 = np.linalg.norm(vec1)
    magnitude_vec2 = np.linalg.norm(vec2)
    
    if magnitude_vec1 == 0 or magnitude_vec2 == 0:
        return 0.0
    
    cosine_sim = dot_product / (magnitude_vec1 * magnitude_vec2)
    return cosine_sim

# Logger class to capture output
class StreamCapture:
    def __init__(self):
        self.output = StringIO()
        self._stdout = sys.stdout

    def __enter__(self):
        sys.stdout = self.output
        return self.output

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._stdout

# Main Function
def score(main_product, main_url, search, logger, log_area):
    data = {}
    
    if search == 'all':
        similar = extract_similar_products(main_product)[:1]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []

            search_functions = [search_google, search_duckduckgo, search_github, search_wikipedia]
            
            for search_func in search_functions:
                futures.append(executor.submit(partial(filtering, search_func(similar), main_product, similar)))

            for future in concurrent.futures.as_completed(futures):
                data[similar] = future.result()

    else:
        similar = extract_similar_products(main_product)[:1]

        for product in similar:

            if search == 'google':
                data[product] = filtering(search_google(product), main_product, product)
            elif search == 'duckduckgo':
                data[product] = filtering(search_duckduckgo(product), main_product, product)
            elif search == 'archive':
                data[product] = filtering(search_archive(product), main_product, product)
            elif search == 'github':
                data[product] = filtering(search_github(product), main_product, product)
            elif search == 'wikipedia':
                data[product] = filtering(search_wikipedia(product), main_product, product)

    logger.write("\n\nFiltered Links ------------------>\n")
    logger.write(str(data) + "\n")
    log_area.text(logger.getvalue())

    logger.write("\n\nCreating Main product Embeddings ---------->\n")
    main_result, main_embedding = get_embeddings(main_url)
    log_area.text(logger.getvalue())

    print("main",main_embedding)
    
    cosine_sim_scores = []

    logger.write("\n\nCreating Similar product Embeddings ---------->\n")
    log_area.text(logger.getvalue())


    for product in data:
        for link in data[product][:2]:

            similar_result, similar_embedding = get_embeddings(link)
            log_area.text(logger.getvalue())

            print(similar_embedding)
            for i in range(len(main_embedding)):
                score = cosine_similarity(main_embedding[i], similar_embedding[i])
                cosine_sim_scores.append((product, link, i, score))
                log_area.text(logger.getvalue())

    logger.write("--------------- DONE -----------------\n")
    log_area.text(logger.getvalue())
    return cosine_sim_scores, main_result

# Streamlit Interface
st.title("Product Infringement Checker")

# Inputs
main_product = st.text_input('Enter Main Product Name', 'Philips led 7w bulb')
main_url = st.text_input('Enter Main Product Manual URL', 'https://www.assets.signify.com/is/content/PhilipsConsumer/PDFDownloads/Colombia/technical-sheets/ODLI20180227_001-UPD-es_CO-Ficha_Tecnica_LED_MR16_Master_7W_Dim_12V_CRI90.pdf')
search_method = st.selectbox('Choose Search Engine', ['duckduckgo', 'google', 'archive', 'github', 'wikipedia', 'all'])

if st.button('Check for Infringement'):
    log_output = st.empty()  # Placeholder for log output

    with st.spinner('Processing...'):
        with StreamCapture() as logger:
            cosine_sim_scores, main_result = score(main_product, main_url, search_method, logger, log_output)

    st.success('Processing complete!')

    st.subheader("Cosine Similarity Scores")

    #  = score(main_product, main_url, search, logger, log_output)
    tags = ['Introduction', 'Specifications', 'Product Overview', 'Safety Information', 'Installation Instructions', 'Setup and Configuration', 'Operation Instructions', 'Maintenance and Care', 'Troubleshooting', 'Warranty Information', 'Legal Information']

    for product, link, index, value in cosine_sim_scores:
        if not index:
            st.write(f"Product: {product}, Link: {link}")
        st.write(f"{tags[index]:<20} - Similarity: {value:.2f}")
import streamlit as st
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor,as_completed
from functools import partial
import numpy as np
from io import StringIO
import sys
import time
from pymongo import MongoClient

# File Imports
from embedding import get_embeddings  # Ensure this file/module is available
from preprocess import filtering  # Ensure this file/module is available
from search import *


# Mongo Connections
srv_connection_uri = "mongodb+srv://adityasm1410:uOh6i11AYFeKp4wd@patseer.5xilhld.mongodb.net/?retryWrites=true&w=majority&appName=Patseer"

client = MongoClient(srv_connection_uri)
db = client['embeddings'] 
collection = db['data']  

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
def score(main_product, main_url, product_count, link_count, search, logger, log_area):

    existing_products_urls = set(collection.distinct('url'))

    data = {}
    similar_products = extract_similar_products(main_product)[:product_count]

    
    # Normal Filtering + Embedding  -----------------------------------------------
    if search == 'All':

        def process_product(product, search_function, main_product):
            search_result = search_function(product)
            return filtering(search_result, main_product, product, link_count)
                
                
        search_functions = {
            'google': search_google,
            'duckduckgo': search_duckduckgo,
            # 'archive': search_archive,
            'github': search_github,
            'wikipedia': search_wikipedia
        }

        with ThreadPoolExecutor() as executor:
            future_to_product_search = {
                executor.submit(process_product, product, search_function, main_product): (product, search_name)
                for product in similar_products
                for search_name, search_function in search_functions.items()
            }

            for future in as_completed(future_to_product_search):
                product, search_name = future_to_product_search[future]
                try:
                    if product not in data:
                        data[product] = {}
                    data[product] = future.result()
                except Exception as e:
                    print(f"Error processing product {product} with {search_name}: {e}")

    else:

        for product in similar_products:

            if search == 'google':
                data[product] = filtering(search_google(product), main_product, product, link_count)
            elif search == 'duckduckgo':
                data[product] = filtering(search_duckduckgo(product), main_product, product, link_count)
            elif search == 'archive':
                data[product] = filtering(search_archive(product), main_product, product, link_count)
            elif search == 'github':
                data[product] = filtering(search_github(product), main_product, product, link_count)
            elif search == 'wikipedia':
                data[product] = filtering(search_wikipedia(product), main_product, product, link_count)


    # Filtered Link -----------------------------------------
    logger.write("\n\nFiltered Links ------------------>\n")
    logger.write(str(data) + "\n")
    log_area.text(logger.getvalue())



    # Main product Embeddings ---------------------------------
    logger.write("\n\nCreating Main product Embeddings ---------->\n")

    # Check main product in MongoDB
    if main_url in existing_products_urls:
        saved_data = collection.find_one({'url': main_url})

        if tag_option not in saved_data:
            main_result , main_embedding = get_embeddings(main_url,tag_option)
        else:
            main_embedding = saved_data[tag_option]
    else:
        main_result , main_embedding = get_embeddings(main_url,tag_option)

    log_area.text(logger.getvalue())
    print("main",main_embedding)

    update_doc = {
        '$set': {
            'product_name': main_product,
            'url': main_url,
            tag_option: main_embedding
            }
    }

    collection.update_one(
        {'url': main_url},
        update_doc,
        upsert=True
    )


    #Similar Products Check            
    cosine_sim_scores = []

    logger.write("\n\nCreating Similar product Embeddings ---------->\n")
    log_area.text(logger.getvalue())


    for product in data:

        if len(data[product])==0:
            logger.write("\n\nNo Product links Found Increase No of Links or Change Search Source\n")
            log_area.text(logger.getvalue())
    
            cosine_sim_scores.append((product,'No Product links Found Increase Number of Links or Change Search Source',None,None))
        
        else:
            for link,present in data[product][:link_count]:
                
                saved_data = collection.find_one({'url': link})

                if present and (tag_option in saved_data):
                    similar_embedding = saved_data[tag_option]
                else:
                    similar_result, similar_embedding = get_embeddings(link,tag_option)

                log_area.text(logger.getvalue())

                print(similar_embedding)
                for i in range(len(main_embedding)):
                    score = cosine_similarity(main_embedding[i], similar_embedding[i])
                    cosine_sim_scores.append((product, link, i, score))
                    log_area.text(logger.getvalue())
                
                update_doc = {
                    '$set': {
                        'product_name': product,
                        'url': link,
                        tag_option: similar_embedding
                    }
                }

                collection.update_one(
                    {'url': link},
                    update_doc,
                    upsert=True
                )

    logger.write("--------------- DONE -----------------\n")
    log_area.text(logger.getvalue())
    return cosine_sim_scores

# Streamlit Interface
st.title("Check Infringement")


# Inputs
main_product = st.text_input('Enter Main Product Name', 'Philips led 7w bulb')
main_url = st.text_input('Enter Main Product Manual URL', 'https://www.assets.signify.com/is/content/PhilipsConsumer/PDFDownloads/Colombia/technical-sheets/ODLI20180227_001-UPD-es_CO-Ficha_Tecnica_LED_MR16_Master_7W_Dim_12V_CRI90.pdf')
search_method = st.selectbox('Choose Search Engine', ['All','duckduckgo', 'google', 'archive', 'github', 'wikipedia'])

col1, col2 = st.columns(2)
with col1:
    product_count = st.number_input("Number of Simliar Products",min_value=1, step=1, format="%i")
with col2:
    link_count = st.number_input("Number of Links per product",min_value=1, step=1, format="%i")


tag_option = st.selectbox('Choose Similarity Method', ["Complete Document Similarity","Field Wise Document Similarity"])


if st.button('Check for Infringement'):
    log_output = st.empty()  # Placeholder for log output

    with st.spinner('Processing...'):
        with StreamCapture() as logger:
            cosine_sim_scores = score(main_product, main_url,product_count, link_count, search_method, logger, log_output)

    st.success('Processing complete!')

    st.subheader("Cosine Similarity Scores")

    #  = score(main_product, main_url, search, logger, log_output)
    if tag_option == 'Complete Document Similarity':
        tags = ['Details']
    else:    
        tags = ['Introduction', 'Specifications', 'Product Overview', 'Safety Information', 'Installation Instructions', 'Setup and Configuration', 'Operation Instructions', 'Maintenance and Care', 'Troubleshooting', 'Warranty Information', 'Legal Information']

    for product, link, index, value in cosine_sim_scores:
        if not index:
            st.write(f"Product: {product}, Link: {link}")
        if value!=None:
            st.write(f"{tags[index]:<20} - Similarity: {value:.2f}")
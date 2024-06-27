import streamlit as st
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor,as_completed
from functools import partial
import numpy as np
from io import StringIO
import sys
import time
import pandas as pd
from pymongo import MongoClient
import plotly.express as px
from pinecone import Pinecone, ServerlessSpec
import chromadb
import requests
from io import BytesIO
from PyPDF2 import PdfReader
import hashlib
import os

# File Imports
from embedding import get_embeddings,get_image_embeddings,get_embed_chroma,imporve_text  # Ensure this file/module is available
from preprocess import filtering  # Ensure this file/module is available
from search import *


# Chroma Connections
client = chromadb.PersistentClient(path = "embeddings")
collection = client.get_or_create_collection(name="data",metadata={"hnsw:space": "l2"})


def generate_hash(content):
    return hashlib.sha256(content.encode('utf-8')).hexdigest()

def get_key(link):
    text = ''
    try:
        # Fetch the PDF file from the URL
        response = requests.get(link)
        response.raise_for_status()  # Raise an error for bad status codes

        # Use BytesIO to handle the PDF content in memory
        pdf_file = BytesIO(response.content)

        # Load the PDF file
        reader = PdfReader(pdf_file)
        num_pages = len(reader.pages)
        
        first_page_text = reader.pages[0].extract_text()
        if first_page_text:
            text += first_page_text
        

        last_page_text = reader.pages[-1].extract_text()
        if last_page_text:
            text += last_page_text

    except requests.exceptions.HTTPError as e:
        print(f'HTTP error occurred: {e}')
    except Exception as e:
        print(f'An error occurred: {e}')
    
    unique_key = generate_hash(text)
    
    return unique_key

# Cosine Similarity Function
def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    dot_product = np.dot(vec1, vec2.T)
    magnitude_vec1 = np.linalg.norm(vec1)
    magnitude_vec2 = np.linalg.norm(vec2)
    
    if magnitude_vec1 == 0 or magnitude_vec2 == 0:
        return 0.0
    
    cosine_sim = dot_product / (magnitude_vec1 * magnitude_vec2)
    return cosine_sim

def update_chroma(product_name,url,key,text,vector,log_area):

    id_list = [key+str(i) for i in range(len(text))]

    metadata_list = [
            {   'key':key,
                'product_name': product_name,
                'url': url,
                'text':item
            }
            for item in text
        ]

    collection.upsert(
        ids = id_list,
        embeddings = vector,
        metadatas = metadata_list
    )

    logger.write(f"\n\u2713 Updated DB - {url}\n\n")
    log_area.text(logger.getvalue())


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


    data = {}
    similar_products = extract_similar_products(main_product)[:product_count]

    print("--> Fetching Manual Links")
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
    logger.write("\n\n\u2713 Filtered Links\n")
    log_area.text(logger.getvalue())


    # Main product Embeddings ---------------------------------
    logger.write("\n\n--> Creating Main product Embeddings\n")

    main_key = get_key(main_url)
    main_text,main_vector = get_embed_chroma(main_url)

    update_chroma(main_product,main_url,main_key,main_text,main_vector,log_area)

    # log_area.text(logger.getvalue())
    print("\n\n\u2713 Main Product embeddings Created")


    logger.write("\n\n--> Creating Similar product Embeddings\n")
    log_area.text(logger.getvalue()) 
    test_embedding = [0]*768 

    for product in data:
        for link in data[product]:

            url, _ = link
            similar_key = get_key(url)

            res = collection.query(
                    query_embeddings = [test_embedding],
                    n_results=1,
                    where={"key": similar_key},
                )

            if not res['distances'][0]:
                similar_text,similar_vector = get_embed_chroma(url)
                update_chroma(product,url,similar_key,similar_text,similar_vector,log_area)


    logger.write("\n\n\u2713 Similar Product embeddings Created\n")
    log_area.text(logger.getvalue())

    top_similar = []

    for idx,chunk in enumerate(main_vector):
        res = collection.query(
                    query_embeddings = [chunk],
                    n_results=1,
                    where={"key": {'$ne':main_key}},
                    include=['metadatas','embeddings','distances']
                )
        
        top_similar.append((main_text[idx],chunk,res,res['distances'][0]))
    
    most_similar_items = sorted(top_similar,key = lambda x:x[3])[:top_similar_count]


    logger.write("--------------- DONE -----------------\n")
    log_area.text(logger.getvalue())

    return most_similar_items





# Streamlit Interface
st.title("Check Infringement")


# Inputs
main_product = st.text_input('Enter Main Product Name', 'Philips led 7w bulb')
main_url = st.text_input('Enter Main Product Manual URL', 'https://www.assets.signify.com/is/content/PhilipsConsumer/PDFDownloads/Colombia/technical-sheets/ODLI20180227_001-UPD-es_CO-Ficha_Tecnica_LED_MR16_Master_7W_Dim_12V_CRI90.pdf')
search_method = st.selectbox('Choose Search Engine', ['All','duckduckgo', 'google', 'archive', 'github', 'wikipedia'])

col1, col2, col3= st.columns(3)
with col1:
    product_count = st.number_input("Number of Simliar Products",min_value=1, step=1, format="%i")
with col2:
    link_count = st.number_input("Number of Links per product",min_value=1, step=1, format="%i")
with col3:
    need_image = st.selectbox("Process Images", ['True','False'])

top_similar_count = st.number_input("Top Similarities to be displayed",value=3,min_value=1, step=1, format="%i")
tag_option = "Complete Document Similarity"


if st.button('Check for Infringement'):
    global log_output  # Placeholder for log output

    tab1, tab2 = st.tabs(["Output", "Console"])

    with tab2:
        log_output = st.empty()

    with tab1:
        with st.spinner('Processing...'):
            with StreamCapture() as logger:
                top_similar_values = score(main_product, main_url, product_count, link_count, search_method, logger, log_output)

        st.success('Processing complete!')

        st.subheader("Cosine Similarity Scores")

        for main_text, main_vector, response, _ in top_similar_values:
            product_name = response['metadatas'][0][0]['product_name']
            link = response['metadatas'][0][0]['url']
            similar_text = response['metadatas'][0][0]['text']

            cosine_score = cosine_similarity([main_vector], response['embeddings'][0])[0][0]

            # Display the product information
            with st.container():
                st.markdown(f"### [Product: {product_name}]({link})")
                st.markdown(f"#### Cosine Score: {cosine_score:.4f}")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Main Text:** {imporve_text(main_text)}")
                with col2:
                    st.markdown(f"**Similar Text:** {imporve_text(similar_text)}")

                st.markdown("---")

    if need_image == 'True':
        with st.spinner('Processing Images...'):
            emb_main = get_image_embeddings(main_product)
            similar_prod = extract_similar_products(main_product)[0]
            emb_similar = get_image_embeddings(similar_prod)

            similarity_matrix = np.zeros((5, 5))
            for i in range(5):
                for j in range(5):
                    similarity_matrix[i][j] = cosine_similarity([emb_main[i]], [emb_similar[j]])[0][0]

            st.subheader("Image Similarity")
            # Create an interactive heatmap
            fig = px.imshow(similarity_matrix,
                            labels=dict(x=f"{similar_prod} Images", y=f"{main_product} Images", color="Similarity"),
                            x=[f"Image {i+1}" for i in range(5)],
                            y=[f"Image {i+1}" for i in range(5)],
                            color_continuous_scale="Viridis")

            # Add title to the heatmap
            fig.update_layout(title="Image Similarity Heatmap")

            # Display the interactive heatmap
            st.plotly_chart(fig)




# main_product = 'Philips led 7w bulb'
# main_url = 'https://www.assets.signify.com/is/content/PhilipsConsumer/PDFDownloads/Colombia/technical-sheets/ODLI20180227_001-UPD-es_CO-Ficha_Tecnica_LED_MR16_Master_7W_Dim_12V_CRI90.pdf'
# search_method = 'duckduckgo'

# product_count = 1
# link_count = 1
# need_image = False


# tag_option = "Field Wise Document Similarity"

# logger = StreamCapture()
# score(main_product, main_url,product_count, link_count, search_method, logger, st.empty())




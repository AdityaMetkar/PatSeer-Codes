import streamlit as st
import requests
import json
import os
import concurrent.futures
import random
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

gemini = ChatGoogleGenerativeAI(model="gemini-1.0-pro-001", google_api_key='AIzaSyBmZtXjJgp7yIAo9joNCZGSxK9PbGMcVaA', temperature=0.1)
gemini1 = ChatGoogleGenerativeAI(model="gemini-1.0-pro-001", google_api_key='AIzaSyABsaDjPujPCBlz4LLxcXDX_bDA9uEL7Xc', temperature=0.1)

def pdf_extractor(link):
    text = ''
    loader = PyPDFLoader(link)
    pages = loader.load_and_split()
    for page in pages:
        text += page.page_content
    return [text]

def web_extractor(link):
    text = ''
    loader = WebBaseLoader(link)
    pages = loader.load_and_split()
    for page in pages:
        text += page.page_content
    return [text]

def feature_extraction(tag, history, context):
    prompt = f'''
    You are an intelligent assistant tasked with updating product information. You have two data sources:
    1. Tag_History: Previously gathered information about the product.
    2. Tag_Context: New data that might contain additional details.

    Your job is to read the Tag_Context and update the relevant field in the Tag_History with any new details found. The field to be updated is the {tag} FIELD.

    Guidelines:
    - Only add new details that are relevant to the {tag} FIELD.
    - Do not add or modify any other fields in the Tag_History.
    - Ensure your response is in coherent sentences, integrating the new details seamlessly into the existing information.

    Here is the data:

    Tag_Context: {str(context)}
    Tag_History: {history}

    Respond with the updated Tag_History.
    '''
    model = random.choice([gemini, gemini1])
    result = model.invoke(prompt)
    return result.content

def main(link): 
    history = {
        "Introduction": "",
        "Specifications": "",
        "Product Overview": "",
        "Safety Information": "",
        "Installation Instructions": "",
        "Setup and Configuration": "",
        "Operation Instructions": "",
        "Maintenance and Care": "",
        "Troubleshooting": "",
        "Warranty Information": "",
        "Legal Information": ""
    }

    # Extract Text
    if link.endswith('.md') or link[8:11] == 'en.':
        text = web_extractor(link)
    else:
        text = pdf_extractor(link)

    # Create Chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=100,
        separators=["", '', " "]
    )
    chunks = text_splitter.create_documents(text)

    for idx, chunk in enumerate(chunks):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_key = {
                executor.submit(feature_extraction, key, history[key], chunk.page_content): key for key in history
            }
            for future in concurrent.futures.as_completed(future_to_key):
                key = future_to_key[future]
                try:
                    response = future.result()
                    history[key] = response
                    st.write(f"Intermediate result for {key}: {response}")
                except Exception as e:
                    st.write(f"Error processing {key}: {e}")

    return history

st.title('Product Information Extractor')
link = st.text_input('Enter the link to the product document:')
if st.button('Process'):
    if link:
        final_result = main(link)
        st.write('Final extracted fields/tags:')
        st.json(final_result)
    else:
        st.write('Please enter a valid link.')

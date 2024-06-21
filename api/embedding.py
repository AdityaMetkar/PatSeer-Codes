from PyPDF2 import PdfReader
import requests
import json
import os
import concurrent.futures
import random
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
from langchain_core.messages import HumanMessage
from io import BytesIO
import numpy as np

from search import search_images

gemini = ChatGoogleGenerativeAI(model="gemini-1.0-pro-001",google_api_key='AIzaSyCo-TeDp0Ou--UwhlTgMwCoTEZxg6-v7wA',temperature = 0.1)
gemini1 = ChatGoogleGenerativeAI(model="gemini-1.0-pro-001",google_api_key='AIzaSyAtnUk8QKSUoJd3uOBpmeBNN-t8WXBt0zI',temperature = 0.1)
gemini2 = ChatGoogleGenerativeAI(model="gemini-1.0-pro-001",google_api_key='AIzaSyBzbZQBffHFK3N-gWnhDDNbQ9yZnZtaS2E',temperature = 0.1)
gemini3 = ChatGoogleGenerativeAI(model="gemini-1.0-pro-001",google_api_key='AIzaSyBNN4VDMAOB2gSZha6HjsTuH71PVV69FLM',temperature = 0.1)

vision = ChatGoogleGenerativeAI(model="gemini-1.5-flash",google_api_key='AIzaSyCo-TeDp0Ou--UwhlTgMwCoTEZxg6-v7wA',temperature = 0.1)
vision1 = ChatGoogleGenerativeAI(model="gemini-1.5-flash",google_api_key='AIzaSyAtnUk8QKSUoJd3uOBpmeBNN-t8WXBt0zI',temperature = 0.1)
vision2 = ChatGoogleGenerativeAI(model="gemini-1.5-flash",google_api_key='AIzaSyBzbZQBffHFK3N-gWnhDDNbQ9yZnZtaS2E',temperature = 0.1)
vision3 = ChatGoogleGenerativeAI(model="gemini-1.5-flash",google_api_key='AIzaSyBNN4VDMAOB2gSZha6HjsTuH71PVV69FLM',temperature = 0.1)


genai.configure(api_key="AIzaSyAtnUk8QKSUoJd3uOBpmeBNN-t8WXBt0zI")

def pdf_extractor(link):
    text = ''

    try:
        # Fetch the PDF file from the URL
        response = requests.get(link)
        response.raise_for_status()  # Raise an error for bad status codes

        # Use BytesIO to handle the PDF content in memory
        pdf_file = BytesIO(response.content)

        # Load the PDF file
        reader = PdfReader(pdf_file)
        for page in reader.pages:
            text += page.extract_text()  # Extract text from each page

    except requests.exceptions.HTTPError as e:
        print(f'HTTP error occurred: {e}')
    except Exception as e:
        print(f'An error occurred: {e}')
    
    return [text]

def web_extractor(link):
    text = ''

    try:
        loader = WebBaseLoader(link)
        pages = loader.load_and_split()

        for page in pages:
            text+=page.page_content
    except:
        pass
    
    return [text]


def feature_extraction(tag, history , context):

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

    model = random.choice([gemini,gemini1,gemini2,gemini3])
    result = model.invoke(prompt)

    return result.content

def feature_extraction_image(url):

    vision = ChatGoogleGenerativeAI(model="gemini-1.5-flash",google_api_key='AIzaSyBzbZQBffHFK3N-gWnhDDNbQ9yZnZtaS2E',temperature = 0.1)
    # result = gemini.invoke('''Hello''')
    # Markdown(result.content)
    # print(result)

    text = 'None'
    message = HumanMessage(content=[
                    {"type": "text", "text": "Please, Describe this image in detail"},
                    {"type": "image_url", "image_url": url}
                ])
    try:
        model = random.choice([vision,vision1,vision2,vision3])
        text = model.invoke([message])
    except:
        return text
    return text.content

def detailed_feature_extraction(find, context):

    prompt = f'''
    You are an intelligent assistant tasked with finding product information. You have one data source and one output format:
    1. Context: The gathered information about the product.
    2. Format: Details which need to be filled based on Context.
    Your job is to read the Context and update the relevant field in Format using Context.
    Guidelines:
    - Only add details that are relevant to the individual FIELD.
    - Do not add or modify any other fields in the Format.
    - If nothing found return None.
    Here is the data:
    The Context is {str(context)}
    The Format is {str(find)}
    '''

    model = random.choice([gemini,gemini1,gemini2,gemini3])
    result = model.invoke(prompt)

    return result.content

def detailed_history(history):

    details = {
    "Introduction": {
        "Product Name": None,
        "Overview of the product": None,
        "Purpose of the manual": None,
        "Audience": None,
        "Additional Details": None
    },
    "Specifications": {
        "Technical specifications": None,
        "Performance metrics": None,
        "Additional Details": None
    },
    "Product Overview": {
        "Product features": None,
        "Key components and parts": None,
        "Additional Details": None
    },
    "Safety Information": {
        "Safety warnings and precautions": None,
        "Compliance and certification information": None,
        "Additional Details": None
    },
    "Installation Instructions": {
        "Unboxing and inventory checklist": None,
        "Step-by-step installation guide": None,
        "Required tools and materials": None,
        "Additional Details": None
    },
    "Setup and Configuration": {
        "Initial setup procedures": None,
        "Configuration settings": None,
        "Troubleshooting setup issues": None,
        "Additional Details": None
    },
    "Operation Instructions": {
        "How to use the product": None,
        "Detailed instructions for different functionalities": None,
        "User interface guide": None,
        "Additional Details": None
    },
    "Maintenance and Care": {
        "Cleaning instructions": None,
        "Maintenance schedule": None,
        "Replacement parts and accessories": None,
        "Additional Details": None
    },
    "Troubleshooting": {
        "Common issues and solutions": None,
        "Error messages and their meanings": None,
        "Support Information": None,
        "Additional Details": None
    },
    "Warranty Information": {
        "Terms and Conditions": None,
        "Service and repair information": None,
        "Additional Details": None
    },
    "Legal Information": {
        "Copyright information": None,
        "Trademarks and patents": None,
        "Disclaimers": None,
        "Additional Details": None

    }
}

    for key,val in history.items():

        find = details[key]

        details[key] = str(detailed_feature_extraction(find,val))

    return details


def get_embeddings(link,tag_option): 

        print(f"\nCreating Embeddings ----- {link}")

        if tag_option=='Complete Document Similarity':
            history = { "Details": "" }

        else:
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

        # Extract Text -----------------------------
        print("Extracting Text")
        if link[-3:] == '.md' or link[8:11] == 'en.':
            text = web_extractor(link)
        else:
            text = pdf_extractor(link)

        # Create Chunks ----------------------------
        print("Writing Tag Data")

        if tag_option=="Complete Document Similarity":
            history["Details"] = feature_extraction("Details", history["Details"], text[0][:50000])
            
        else:
            chunks = text_splitter.create_documents(text)

            for chunk in chunks:

                with concurrent.futures.ThreadPoolExecutor() as executor:
                        future_to_key = {
                            executor.submit(
                                feature_extraction, f"Product {key}", history[key], chunk.page_content
                            ): key for key in history
                        }
                        for future in concurrent.futures.as_completed(future_to_key):
                            key = future_to_key[future]
                            try:
                                response = future.result()
                                history[key] = response
                            except Exception as e:
                                print(f"Error processing {key}: {e}")
            
        print("Creating Vectors")
        genai_embeddings=[]
            
        for tag in history:
            result = genai.embed_content(
                    model="models/embedding-001",
                    content=history[tag],
                    task_type="retrieval_document")
            genai_embeddings.append(result['embedding'])


        return history,np.array(genai_embeddings)

def get_image_embeddings(Product):
    image_embeddings = []
    
    links = search_images(Product)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        descriptions = list(executor.map(feature_extraction_image, links))
    
    for description in descriptions:
        result = genai.embed_content(
                model="models/embedding-001",
                content=description,
                task_type="retrieval_document")
        
        image_embeddings.append(result['embedding'])
    # print(image_embeddings)
    return image_embeddings


            
global text_splitter
global data
global history


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 10000,
    chunk_overlap  = 100,
    separators = ["",''," "]
)

if __name__ == '__main__':
    # print(get_embeddings('https://www.galaxys24manual.com/wp-content/uploads/pdf/galaxy-s24-manual-SAM-S921-S926-S928-OS14-011824-FINAL-US-English.pdf',"Complete Document Similarity"))
    print(get_image_embeddings(Product='Samsung Galaxy S24'))
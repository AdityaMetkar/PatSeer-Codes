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


gemini = ChatGoogleGenerativeAI(model="gemini-1.0-pro-001",google_api_key='AIzaSyBmZtXjJgp7yIAo9joNCZGSxK9PbGMcVaA',temperature = 0.1)
gemini1 = ChatGoogleGenerativeAI(model="gemini-1.0-pro-001",google_api_key='AIzaSyABsaDjPujPCBlz4LLxcXDX_bDA9uEL7Xc',temperature = 0.1)
gemini2 = ChatGoogleGenerativeAI(model="gemini-1.0-pro-001",google_api_key='AIzaSyBCIQgt1uK7-sJH5Afg5vUZ99EWkx5gSU0',temperature = 0.1)
gemini3 = ChatGoogleGenerativeAI(model="gemini-1.0-pro-001",google_api_key='AIzaSyBot9W5Q-BKQ66NAYRUmVeloXWEbXOXTmM',temperature = 0.1)

genai.configure(api_key="AIzaSyBmZtXjJgp7yIAo9joNCZGSxK9PbGMcVaA")


def pdf_extractor(link):
    text = ''

    try:
        loader = PyPDFLoader(link)
        pages = loader.load_and_split()

        for page in pages:
            text+=page.page_content
    except:
        pass
    
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

    model = random.choice([gemini,gemini1])
    result = model.invoke(prompt)

    return result.content

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


def get_embeddings(link): 

        print(f"\nCreating Embeddings ----- {link}")
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
            
            # history = detailed_history(history)
        print("Creating Vectors")
        genai_embeddings=[]

        for tag in history:
            try:    
                result = genai.embed_content(
                        model="models/embedding-001",
                        content=history[tag],
                        task_type="retrieval_document")
                genai_embeddings.append(result['embedding'])
            except:
                genai_embeddings.append([0]*768)


        return history,genai_embeddings
            
global text_splitter
global data
global history


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 10000,
    chunk_overlap  = 100,
    separators = ["",''," "]
)


if __name__ == '__main__':
    pass

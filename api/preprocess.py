import requests
import json
import random
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from langchain_community.document_loaders import PyPDFLoader
from langdetect import detect_langs
import requests
from PyPDF2 import PdfReader
from io import BytesIO
from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import ChatGoogleGenerativeAI
import logging
from pymongo import MongoClient


# Mongo Connections
srv_connection_uri = "mongodb+srv://adityasm1410:uOh6i11AYFeKp4wd@patseer.5xilhld.mongodb.net/?retryWrites=true&w=majority&appName=Patseer"

client = MongoClient(srv_connection_uri)
db = client['embeddings'] 
collection = db['data']  


# API Urls -----

# main_url = "http://127.0.0.1:5000/search/all"
main_url = "http://127.0.0.1:8000/search/all"
# main_product = "Samsung Galaxy s23 ultra"

# Revelevance Checking Models -----
gemini = ChatGoogleGenerativeAI(model="gemini-1.0-pro-001",google_api_key='AIzaSyCo-TeDp0Ou--UwhlTgMwCoTEZxg6-v7wA',temperature = 0.1)
gemini1 = ChatGoogleGenerativeAI(model="gemini-1.0-pro-001",google_api_key='AIzaSyAtnUk8QKSUoJd3uOBpmeBNN-t8WXBt0zI',temperature = 0.1)
gemini2 = ChatGoogleGenerativeAI(model="gemini-1.0-pro-001",google_api_key='AIzaSyBzbZQBffHFK3N-gWnhDDNbQ9yZnZtaS2E',temperature = 0.1)
gemini3 = ChatGoogleGenerativeAI(model="gemini-1.0-pro-001",google_api_key='AIzaSyBNN4VDMAOB2gSZha6HjsTuH71PVV69FLM',temperature = 0.1)


API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-xxl"
headers = {"Authorization": "Bearer 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"}

# Error Debug
logging.basicConfig(level=logging.INFO)



# Global Var --------

data = False 
seen = set()
existing_products_urls = set(collection.distinct('url'))



def get_links(main_product,api_key):
    params = {
        "API_KEY": f"{api_key}",
        "product": f"{main_product}",
    }

    # Flask
    response = requests.get(main_url, params=params)

    # FastAPI
    # response = requests.post(main_url, json=params)


    if response.status_code == 200:
        results = response.json()
        with open('data.json', 'w') as f:
            json.dump(results, f)
    else:
        print(f"Failed to fetch results: {response.status_code}")



def language_preprocess(text):
    try:
        if detect_langs(text)[0].lang == 'en':
            return True
        return False
    except:
        return False


def relevant(product, similar_product, content):

    try:
        payload = { "inputs": f'''Do you think that the given content is similar to {similar_product} and {product}, just Respond True or False  \nContent for similar product:  {content}'''}
        
        # response = requests.post(API_URL, headers=headers, json=payload)
        # output = response.json()
        # return bool(output[0]['generated_text'])
            
        model = random.choice([gemini,gemini1,gemini2,gemini3])
        result = model.invoke(f'''Do you think that the given content is similar to {similar_product} and {product}, just Respond True or False  \nContent for similar product:  {content}''')
        return bool(result)

    except:
        return False
        
        

def download_pdf(url, timeout=10):
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return BytesIO(response.content)

    except requests.RequestException as e:
        logging.error(f"PDF download error: {e}")
        return None

def extract_text_from_pdf(pdf_file, pages):
    reader = PdfReader(pdf_file)
    extracted_text = ""

    l = len(reader.pages)
    
    try:
        for page_num in pages:
            if page_num < l:
                page = reader.pages[page_num]
                extracted_text += page.extract_text() + "\n"
            else:
                print(f"Page {page_num} does not exist in the document.")
        
        return extracted_text
    
    except:
        return 'हे चालत नाही'
    
def extract_text_online(link):

    loader = WebBaseLoader(link)
    pages = loader.load_and_split()

    text = ''

    for page in pages[:3]:
        text+=page.page_content
    
    return text


def process_link(link, main_product, similar_product):
    if link in seen:
        return None
    seen.add(link)
    try:
        if link[-3:]=='.md' or link[8:11] == 'en.':
            text = extract_text_online(link)
        else:
            pdf_file = download_pdf(link)
            text = extract_text_from_pdf(pdf_file, [0, 2, 4])

        if language_preprocess(text):
            if relevant(main_product, similar_product, text):
                print("Accepted",link)
                return link
    except:
        pass
    print("NOT Accepted",link)
    return None

def filtering(urls, main_product, similar_product, link_count):
    res = []

    # print(f"Filtering Links of ---- {similar_product}")
    # Main Preprocess ------------------------------
    # with ThreadPoolExecutor() as executor:
    #     futures = {executor.submit(process_link, link, main_product, similar_product): link for link in urls}
    #     for future in concurrent.futures.as_completed(futures):
    #         result = future.result()
    #         if result is not None:
    #             res.append(result)

    # return res

    count = 0

    print(f"Filtering Links of ---- {similar_product}")

    for link in urls:

        if link in existing_products_urls:
            res.append((link,1))
            count+=1
        
        else:
            result = process_link(link, main_product, similar_product)
        
            if result is not None:
                res.append((result,0))
                count += 1
        
        if count == link_count:
            break

    return res


# Main Functions -------------------------------------------------->

# get_links()
# preprocess()


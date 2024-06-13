from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup
from googlesearch import search
from duckduckgo_search import DDGS
import concurrent.futures
import re

app = FastAPI()

API_KEY_DEFAULT = '12345'

class SearchRequest(BaseModel):
    API_KEY: str
    product: str

# Function to search DuckDuckGo
def duckduckgo_search(query):
    try:
        results = DDGS().text(f"{query} manual filetype:pdf", max_results=5)
        return [res['href'] for res in results]
    except:
        return []

# Function to search Google
def google_search(query):
    links = []
    try:
        api_key = 'AIzaSyDV_uJwrgNtawqtl6GDfeUj6NqO-H1tA4c'
        search_engine_id = 'c4ca951b9fc6949cb'
        
        url = f"https://www.googleapis.com/customsearch/v1"
        params = {
            "key": api_key,
            "cx": search_engine_id,
            "q": query + " manual filetype:pdf"
        }

        response = requests.get(url, params=params)
        results = response.json()

        for item in results.get('items', []):
            links.append(item['link'])
    except:
        pass
    
    try:
        extension = "ext:pdf"
        for result in search(query + " manual " + extension, num_results=5):
            if result.endswith('.pdf'):
                links.append(result)
    except:
        pass

    return links

# Function to search Internet Archive
def archive_search(query):
    try:
        url = "https://archive.org/advancedsearch.php"
        params = {
            'q': f'{query} manual',
            'fl[]': ['identifier', 'title', 'format'],
            'rows': 50,
            'page': 1,
            'output': 'json'
        }

        response = requests.get(url, params=params)
        data = response.json()

        def extract_hyperlinks(url):
            response = requests.get(url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    if href.endswith('.pdf'):
                        pdf_files.append(url + '/' + href)
                    if href.endswith('.iso'):
                        extract_pdf_from_iso(url + '/' + href + '/')

        def extract_pdf_from_iso(iso_url):
            iso_response = requests.get(iso_url)
            if iso_response.status_code == 200:
                iso_soup = BeautifulSoup(iso_response.text, 'html.parser')
                for link in iso_soup.find_all('a', href=True):
                    href = link['href']
                    if href.endswith('.pdf'):
                        pdf_files.append('https:' + href)

        pdf_files = []

        def process_doc(doc):
            identifier = doc.get('identifier', 'N/A')
            pdf_link = f"https://archive.org/download/{identifier}"
            extract_hyperlinks(pdf_link)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_doc, doc) for doc in data['response']['docs']]
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    print(f'Generated an exception: {exc}')

        return pdf_files
    
    except:
        return []

def github_search(query):
    try:
        url = f"https://api.github.com/search/code?q={query}+extension:md"
        headers = {
            'Authorization': 'Token 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'
        }
        response = requests.get(url, headers=headers)
        data = response.json()
        links = [item['html_url'].replace('/blob','').replace('//github','//raw.github') for item in data['items']]
        return links
    
    except:
        return []

def extract_similar_products(query):
    results = DDGS().chat(f'{query} Similar Products')
    pattern = r'^\d+\.\s(.+)$'
    matches = re.findall(pattern, results, re.MULTILINE)
    matches = [item.split(': ')[0] for item in matches]
    return matches[:5] if matches else []

@app.get('/')
def read_root():
    return {"message": "Welcome to the search API"}

@app.post('/search/google')
async def search_google(request: SearchRequest):
    if request.API_KEY == API_KEY_DEFAULT:
        results = {request.product: google_search(request.product)}
        similar_products = extract_similar_products(request.product)
        for p in similar_products:
            results[p] = google_search(p)
        return results
    else:
        raise HTTPException(status_code=401, detail="Invalid API key")

@app.post('/search/duckduckgo')
async def search_duckduckgo(request: SearchRequest):
    if request.API_KEY == API_KEY_DEFAULT:
        results = {request.product: duckduckgo_search(request.product)}
        similar_products = extract_similar_products(request.product)
        for p in similar_products:
            results[p] = duckduckgo_search(p)
        return results
    else:
        raise HTTPException(status_code=401, detail="Invalid API key")

@app.post('/search/archive')
async def search_archive(request: SearchRequest):
    if request.API_KEY == API_KEY_DEFAULT:
        results = {request.product: archive_search(request.product)}
        similar_products = extract_similar_products(request.product)
        
        def process_product(product):
            return product, archive_search(product)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_product = {executor.submit(process_product, p): p for p in similar_products}
            for future in concurrent.futures.as_completed(future_to_product):
                product, result = future.result()
                results[product] = result

        return results
    else:
        raise HTTPException(status_code=401, detail="Invalid API key")

@app.post('/search/github')
async def search_github(request: SearchRequest):
    if request.API_KEY == API_KEY_DEFAULT:
        results = {request.product: github_search(request.product)}
        similar_products = extract_similar_products(request.product)
        for p in similar_products:
            results[p] = github_search(p)
        return results
    else:
        raise HTTPException(status_code=401, detail="Invalid API key")

@app.post('/search/all')
async def search_all(request: SearchRequest):
    if request.API_KEY == API_KEY_DEFAULT:
        results = {
            request.product: [
                {'duckduckgo': duckduckgo_search(request.product)},
                {'google': google_search(request.product)},
                {'github': github_search(request.product)},
                {'archive': archive_search(request.product)}
            ]
        }
        
        def search_product(p):
            return {
                'product': p,
                'duckduckgo': duckduckgo_search(p),
                'google': google_search(p),
                'github': github_search(p),
                'archive': archive_search(p)
            }

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_product = {executor.submit(search_product, p): p for p in extract_similar_products(request.product)}
            for future in concurrent.futures.as_completed(future_to_product):
                result = future.result()
                product = result['product']
                results[product] = [
                    {'duckduckgo': result['duckduckgo']},
                    {'google': result['google']},
                    {'github': result['github']},
                    {'archive': result['archive']}
                ]

        return results
    else:
        raise HTTPException(status_code=401, detail="Invalid API key")
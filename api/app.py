from flask import Flask, request, jsonify, render_template
import requests
from bs4 import BeautifulSoup
from googlesearch import search
from duckduckgo_search import DDGS
import concurrent.futures
import re


app = Flask(__name__)

API_KEY_DEFAULT = '12345'

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

        # Make the request
        response = requests.get(url, params=params)
        data = response.json()

        # Function to extract hyperlinks from a webpage
        def extract_hyperlinks(url):        
            # Send a GET request to the URL
            response = requests.get(url)
            
            # Check if the request was successful
            if response.status_code == 200:
                # Parse the HTML content of the page
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Find all <a> tags (hyperlinks)
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    if href.endswith('.pdf'):
                        pdf_files.append(url+'/'+href)
                    if href.endswith('.iso'):
                        # If the link ends with .iso, follow the link and extract .pdf hyperlinks
                        extract_pdf_from_iso(url+'/'+href+'/')

        # Function to extract .pdf hyperlinks from an .iso file
        def extract_pdf_from_iso(iso_url):        
            # Send a GET request to the ISO URL
            iso_response = requests.get(iso_url)
            
            # Check if the request was successful
            if iso_response.status_code == 200:
                # Parse the HTML content of the ISO page
                iso_soup = BeautifulSoup(iso_response.text, 'html.parser')
                
                # Find all <a> tags (hyperlinks) in the ISO page
                for link in iso_soup.find_all('a', href=True):
                    href = link['href']
                    if href.endswith('.pdf'):
                        pdf_files.append('https:'+href)

        pdf_files = []

        def process_doc(doc):
            identifier = doc.get('identifier', 'N/A')
            # title = doc.get('title', 'N/A')
            # format = doc.get('format', 'N/A')
            pdf_link = f"https://archive.org/download/{identifier}"
            extract_hyperlinks(pdf_link)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_doc, doc) for doc in data['response']['docs']]

            # Optionally, wait for all futures to complete and handle any exceptions
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()  # This will raise an exception if the function call raised
                except Exception as exc:
                    print(f'Generated an exception: {exc}')


        return pdf_files
    
    except:
        return []

def github_search(query):

    try:
        # GitHub Search API endpoint
        url = f"https://api.github.com/search/code?q={query}+extension:md"

        headers = {
        'Authorization': 'Token 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'
        }

        # Make the request
        response = requests.get(url,headers=headers)
        data = response.json()
        links = [item['html_url'].replace('/blob','').replace('//github','//raw.github') for item in data['items']]

        return links
    
    except:
        return []
    

#Similarity Check 

def extract_similar_products(query):
    results = DDGS().chat(f'{query} Similar Products')

    pattern = r'^\d+\.\s(.+)$'
    matches = re.findall(pattern, results, re.MULTILINE)
    matches = [item.split(': ')[0] for item in matches]
    print(matches)

    return matches[:5] if matches else []


# Define API routes -------------------------------------------------------

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/search/google', methods=['GET','POST'])
def search_google():
    
    if request.method == 'POST':
        data = request.get_json()
        api_key = data.get('API_KEY')
        product = data.get('product')
    else:
        product = request.args.get('product')
        api_key = request.args.get('API_KEY')

    similar_products = extract_similar_products(product)
    if api_key == API_KEY_DEFAULT:
        results = {product: google_search(product)}
        for p in similar_products:
            results[p] = google_search(p)
        return jsonify(results)

    else:
        return jsonify({'error': 'Invalid API key'}), 401

@app.route('/search/duckduckgo', methods=['GET','POST'])
def search_duckduckgo():

    if request.method == 'POST':
        data = request.get_json()
        api_key = data.get('API_KEY')
        product = data.get('product')
    else:
        product = request.args.get('product')
        api_key = request.args.get('API_KEY')
    
    similar_products = extract_similar_products(product)


    if api_key == API_KEY_DEFAULT:
        results = {product: duckduckgo_search(product)}
        for p in similar_products:
            results[p] = duckduckgo_search(p)
        return jsonify(results)
    else:
        return jsonify({'error': 'Invalid API key'}), 401


@app.route('/search/archive', methods=['GET','POST'])
def search_archive():

    if request.method == 'POST':
        data = request.get_json()
        api_key = data.get('API_KEY')
        product = data.get('product')
    else:
        product = request.args.get('product')
        api_key = request.args.get('API_KEY')
    
    # Retrieve custom headers if any

    similar_products = extract_similar_products(product)

    if api_key == API_KEY_DEFAULT:
        results = {product: archive_search(product)}
        
        def process_product(product):
            return product, archive_search(product)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Map the process_product function to similar_products
            future_to_product = {executor.submit(process_product, p): p for p in similar_products}

            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_product):
                product, result = future.result()
                results[product] = result

        return jsonify(results)

    else:
        return jsonify({'error': 'Invalid API key'}), 401


@app.route('/search/github', methods=['GET','POST'])
def search_github():
        
    if request.method == 'POST':
        data = request.get_json()
        api_key = data.get('API_KEY')
        product = data.get('product')
    else:
        product = request.args.get('product')
        api_key = request.args.get('API_KEY')

    similar_products = extract_similar_products(product)

    if api_key == API_KEY_DEFAULT:
        results = {product: github_search(product)}
        for p in similar_products:
            results[p] = github_search(p)
        return jsonify(results)

    else:
        return jsonify({'error': 'Invalid API key'}), 401


@app.route('/search/all', methods=['GET','POST'])
def search_all():

    if request.method == 'POST':
        data = request.get_json()
        api_key = data.get('API_KEY')
        product = data.get('product')
    else:
        product = request.args.get('product')
        api_key = request.args.get('API_KEY')

    similar_products = extract_similar_products(product)
    

    if api_key == API_KEY_DEFAULT:

        results = {
            product : [{'duckduckgo': duckduckgo_search(product)},{'google': google_search(product)},{'github': github_search(product)},{'archive': archive_search(product)}]
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
            future_to_product = {executor.submit(search_product, p): p for p in similar_products}
            
            for future in concurrent.futures.as_completed(future_to_product):
                result = future.result()
                product = result['product']
                results[product] = [
                    {'duckduckgo': result['duckduckgo']},
                    {'google': result['google']},
                    {'github': result['github']},
                    {'archive': result['archive']}
                ]

        return jsonify(results)

    else:
        return jsonify({'error': 'Invalid API key'}), 401

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)


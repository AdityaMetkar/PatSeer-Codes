
## Infringement Checker for Manuals/Patents - Documentation

### Objective
The objective of this Streamlit application is to compare a user-provided main product manual or patent with similar documents found on the internet. This comparison is based on embedding vectors and cosine similarity scores.

### Components and Functionality

#### 1. **Streamlit Interface**
- **Inputs:**
  - **Main Product Name:** Text input where the user specifies the name of the main product or patent.
  - **Main Product Manual URL:** Text input where the user provides the URL to the PDF manual or patent document of the main product (Manual needs to be of same product mentioned above).
  - **Search Engine:** Dropdown to choose the source(s) (e.g., Google, DuckDuckGo, etc.) to search for similar documents.
  - **Number of Similar Products:** Number input to specify how many similar products or patents to search for.
  - **Number of Links per Product:** Number input to specify how many links (documents) to consider per similar product or patent.
  - **Similarity Method:** Dropdown to choose between 'Complete Document Similarity' (overall document similarity) or 'Field Wise Document Similarity' (section-wise similarity based on predefined tags like Introduction , Product Specification etc...).

- **Functionality:**
  - Upon clicking the "Check for Infringement" button, the application initiates a search and similarity analysis process.
  - Results are displayed in terms of cosine similarity scores between embeddings of sections (or the entire document) of the main product and each link(manual) of similar product found.

## `App.py` Documentation
```python

# Importing custom modules/functions
from embedding import get_embeddings  # Function to get embeddings from documents
from preprocess import filtering  # Function to filter search results
from search import *  # Functions to search for similar documents

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


# Main Function
def score(main_product, main_url, product_count, link_count, search, logger, log_area):
    data = {}
    similar_products = extract_similar_products(main_product)[:product_count]
    
    if search == 'all':
        def process_product(product, search_function, main_product):
            search_result = search_function(product)
            return filtering(search_result, main_product, product)
        
        search_functions = {
            'google': search_google,
            'duckduckgo': search_duckduckgo,
            'archive': search_archive,
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
                data[product] = filtering(search_google(product), main_product, product)
            elif search == 'duckduckgo':
                data[product] = filtering(search_duckduckgo(product), main_product, product)
            elif search == 'archive':
                data[product] = filtering(search_archive(product), main_product, product)
            elif search == 'github':
                data[product] = filtering(search_github(product), main_product, product)
            elif search == 'wikipedia':
                data[product] = filtering(search_wikipedia(product), main_product, product)



    main_result, main_embedding = get_embeddings(main_url, tag_option)

    cosine_sim_scores = []


    for product in data:
        for link in data[product][:link_count]:
            similar_result, similar_embedding = get_embeddings(link, tag_option)

            for i in range(len(main_embedding)):
                score = cosine_similarity(main_embedding[i], similar_embedding[i])
                cosine_sim_scores.append((product, link, i, score))

    return cosine_sim_scores, main_result
```

#### 3. **Streamlit Interface Implementation**
```python
# Streamlit Interface
st.title("Check Infringement")

# Inputs
main_product = st.text_input('Enter Main Product Name', 'Product Name')
main_url = st.text_input('Enter Main Product Manual URL', 'Link')
search_method = st.selectbox('Choose Search Engine', ['duckduckgo', 'google', 'archive', 'github', 'wikipedia', 'all'])

col1, col2 = st.columns(2)
with col1:
    product_count = st.number_input("Number of Similar Products", min_value=1, step=1, format="%i")
with col2:
    link_count = st.number_input("Number of Links per Product", min_value=1, step=1, format="%i")

tag_option = st.selectbox('Choose Similarity Method', ["Complete Document Similarity", "Field Wise Document Similarity"])

```



## Module Overview `search.py`

### Purpose
The `search.py` module provides functions to search for PDF documents related to a given product or topic from various sources on the internet. These functions are integrated into the above discussed application designed to compare a main product manual or patent with similar documents found online, using embedding vectors and cosine similarity scores.

### Functions and Usage

#### 1. `search_duckduckgo(query)`
- **Purpose:** Searches DuckDuckGo for PDF documents related to the given `query`.
- **Parameters:**
  - `query`: A string representing the product or topic to search for.
- **Returns:** 
  - A list of URLs pointing to PDF documents related to the query.

 
```python
from duckduckgo_search import DDGS

def search_duckduckgo(query):
    try:
        results = DDGS().text(f"{query} manual filetype:pdf", max_results=5)
        return [res['href'] for res in results]
    except:
        return []
```

#### 2. `search_google(query)`
- **Purpose:** Searches Google for PDF documents related to the given `query`.
- **Parameters:**
  - `query`: A string representing the product or topic to search for.
- **Returns:** 
  - A list of URLs pointing to PDF documents related to the query.
  
```python
def search_google(query):

    links = []
    try:
        api_key = 'YOUR_GOOGLE_API_KEY'
        search_engine_id = 'YOUR_CUSTOM_SEARCH_ENGINE_ID'
        
        url = f"https://www.googleapis.com/customsearch/v1"
        params = {
            "key": api_key,
            "cx": search_engine_id,
            "q": query + " manual filetype:pdf"
        }

        response = requests.get(url, params=params)
        urls = response.json()
```

#### 3. `search_archive(query)`
- **Purpose:** Searches the Internet Archive for PDF documents related to the given `query`.
- **Parameters:**
  - `query`: A string representing the product or topic to search for.
- **Returns:** 
  - A list of URLs pointing to PDF documents related to the query.
  
```python

def search_archive(query):

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
        urls = response.json()

```

#### 4. `search_github(query)`
- **Purpose:** Searches GitHub for Markdown files related to the given `query`.
- **Parameters:**
  - `query`: A string representing the product or topic to search for.
- **Returns:** 
  - A list of URLs pointing to Markdown files related to the query.
  
```python
import requests

def search_github(query):

    try:
        url = f"https://api.github.com/search/code?q={query}+extension:md"

        headers = {
        'Authorization': 'Token YOUR_GITHUB_TOKEN'
        }

        response = requests.get(url,headers=headers)
        data = response.json()
        urls = [item['html_url'] for item in data['items']]

```

#### 5. `search_wikipedia(product)`
- **Purpose:** Searches Wikipedia for articles related to the given `product`.
- **Parameters:**
  - `product`: A string representing the product or topic to search for.
- **Returns:** 
  - A list of URLs pointing to Wikipedia articles related to the product.
  
```python

def search_wikipedia(product):

    api_url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "opensearch",
        "search": product,
        "limit": 5,
        "namespace": 0,
        "format": "json"
    }
    
    response = requests.get(api_url, params=params)
    urls = response.json()

```

### Integration
These functions are used within the `score` function of the main application to retrieve links to PDF documents or relevant articles from the internet. They are called based on the user's selected search engine (`duckduckgo`, `google`, `archive`, `github`, `wikipedia`) or a combination (`all`) to find similar documents to compare against the main product manual or patent.

#### Summary
The `search.py` module provides essential functions for searching multiple sources on the internet to find documents related to a specified product or topic. These functions enhance the capability of the application to gather a comprehensive set of similar documents for comparison, aiding in the detection of infringement based on document content and structure.

---

### Module Overview `preprocess.py`

This Module is designed to perform several tasks related to fetching, processing, and analyzing content from various sources (like PDFs and web pages) related to a given product. It utilizes APIs and generative AI models for relevance checking and language processing.


### API URLs and Models

Used Multiple APIs so that the Processing speeds up as each API is Limited with 60 req/min.

```python

# Initialize models for relevance checking -----
gemini = ChatGoogleGenerativeAI(model="gemini-1.0-pro-001", google_api_key='YOUR_API_KEY', temperature=0.1)
gemini1 = ChatGoogleGenerativeAI(model="gemini-1.0-pro-001", google_api_key='YOUR_API_KEY', temperature=0.1)
gemini2 = ChatGoogleGenerativeAI(model="gemini-1.0-pro-001", google_api_key='YOUR_API_KEY', temperature=0.1)
gemini3 = ChatGoogleGenerativeAI(model="gemini-1.0-pro-001", google_api_key='YOUR_API_KEY', temperature=0.1)

# Hugging Face API for language model -----
API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-xxl"
headers = {"Authorization": "Bearer YOUR_HUGGING_FACE_TOKEN"}
```

- `main_url`: URL endpoint for fetching search results related to a product.
- `gemini`, `gemini1`, `gemini2`, `gemini3`: Instances of `ChatGoogleGenerativeAI` models initialized with different API keys for relevance checking.
- `API_URL`: URL for Hugging Face model API.
- `headers`: Authorization headers for accessing the Hugging Face API.


### Function: `get_links(main_product, api_key)`

```python
def get_links(main_product, api_key):

    params = {"API_KEY": api_key, "product": main_product}
    response = requests.get(main_url, params=params)
    if response.status_code == 200:
        results = response.json()
        with open('data.json', 'w') as f:
            json.dump(results, f)
    else:
        print(f"Failed to fetch results: {response.status_code}")
```

- **Purpose**: Fetches URLs related to a main product from an API endpoint and stores results in a JSON file.
- **Parameters**:
  - `main_product`: Name of the main product to fetch related links.
  - `api_key`: API key for accessing the endpoint.

### Function: `language_preprocess(text)`

```python
def language_preprocess(text):
    try:
        if detect_langs(text)[0].lang == 'en':
            return True
        return False
    except:
        return False
```

- **Purpose**: Determines if the detected language of the text is English.
- **Parameters**:
  - `text`: Input text to detect language.
- **Returns**: `True` if the language is English, `False` otherwise.

### Function: `relevant(product, similar_product, content)`

```python
def relevant(product, similar_product, content):

    try:
        payload = {"inputs": f"Do you think that the given content is similar to {similar_product} and {product}, just Respond True or False  \nContent for similar product:  {content}"}
        
        # Alternatively HF Model Can be used.
        # Placeholder for actual API call to Hugging Face model
        # response = requests.post(API_URL, headers=headers, json=payload)
        # output = response.json()
        # return bool(output[0]['generated_text'])
            
        model = random.choice([gemini, gemini1, gemini2, gemini3])
        result = model.invoke(f"Do you think that the given content is similar to {similar_product} and {product}, just Respond True or False  \nContent for similar product:  {content}")
        return bool(result)
    except:
        return False
```

- **Purpose**: Uses a generative AI model to determine the relevance of content to specified products.
- **Parameters**:
  - `product`: Main product name.
  - `similar_product`: Similar product name for comparison.
  - `content`: Text content to evaluate relevance.
- **Returns**: `True` if the content is relevant, `False` otherwise.

### Function: `download_pdf(url, timeout=10)`

```python
def download_pdf(url, timeout=10):
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return BytesIO(response.content)
    except requests.RequestException as e:
        logging.error(f"PDF download error: {e}")
        return None
```

- **Purpose**: Downloads a PDF file from a given URL.
- **Parameters**:
  - `url`: URL of the PDF file.
  - `timeout`: Timeout period for the request (default is 10 seconds).
- **Returns**: `BytesIO` object containing PDF content if successful, `None` otherwise.

### Function: `extract_text_from_pdf(pdf_file, pages)`

```python
def extract_text_from_pdf(pdf_file, pages):
    reader = PdfReader(pdf_file)
    extracted_text = ""
    try:
        for page_num in pages:
            if page_num < len(reader.pages):
                page = reader.pages[page_num]
                extracted_text += page.extract_text() + "\n"
            else:
                print(f"Page {page_num} does not exist in the document.")
        return extracted_text
    except:
        return ""
```

- **Purpose**: Extracts text from specific pages of a PDF file.
- **Parameters**:
  - `pdf_file`: `BytesIO` object containing PDF content.
  - `pages`: List of page numbers to extract text from.
- **Returns**: Extracted text from specified pages as a string.

### Function: `extract_text_online(link)`

```python
def extract_text_online(link):
    loader = WebBaseLoader(link)
    pages = loader.load_and_split()
    
    text = ''
    for page in pages[:3]:
        text += page.page_content
    
    return text
```

- **Purpose**: Extracts text content from an online source (web page).
- **Parameters**:
  - `link`: URL of the web page.
- **Returns**: Extracted text content from the web page as a string.

### Function: `process_link(link, main_product, similar_product)`

```python
def process_link(link, main_product, similar_product):
    if link in seen:
        return None
    seen.add(link)
    
    try:
        if link.endswith('.md') or link.startswith('en.'):
            text = extract_text_online(link)

        else:
            pdf_file = download_pdf(link)
            text = extract_text_from_pdf(pdf_file, [0, 2, 4])

        if language_preprocess(text):
            if relevant(main_product, similar_product, text):
                print("Accepted", link)
                return link
    except:
        pass
    
    print("NOT Accepted", link)
    return None
```

- **Purpose**: Processes a given link to determine its relevance to specified products.
- **Parameters**:
  - `link`: URL of the content to process.
  - `main_product`: Main product name.
  - `similar_product`: Similar product name for comparison.
- **Returns**: Returns the link if it passes relevance checks, `None` otherwise.

### Function: `filtering(urls, main_product, similar_product)`

```python
def filtering(urls, main_product, similar_product):
    res = []
    
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_link, link, main_product, similar_product): link for link in urls}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result is not None:
                res.append(result)
    
    return res
```

- **Purpose**: Filters a list of URLs to find those relevant to specified products using concurrent processing.
- **Parameters**:
  - `urls`: List of URLs to filter.
  - `main_product`: Main product name.
  - `similar_product`: Similar product name for comparison.
- **Returns**: List of URLs that are relevant to the specified products.


#### Summary

This module integrates various functionalities including fetching data from APIs, downloading and processing PDFs, extracting text from web pages, and using AI models for content relevance assessment. It's structured to handle multiple concurrent tasks efficiently using thread pools. Adjustments are needed for specific API keys and model configurations as per deployment requirements.

---

### Module Overview `embedding.py`

The `embedding.py` module is designed to extract text from documents (PDFs or web pages), divide them into chunks, and generate embeddings using generative AI models for each chunk. These embeddings are then used for retrieval or comparison purposes.


### Global Variables

```python

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=10000,
    chunk_overlap=100,
    separators=["", "", " "]
)
```

- **`text_splitter`**: Instance of `RecursiveCharacterTextSplitter` configured for splitting text into manageable chunks.

### Function: `pdf_extractor(link)`

```python
def pdf_extractor(link):

    text = ''

    try:
        response = requests.get(link)
        response.raise_for_status()

        pdf_file = BytesIO(response.content)
        reader = PdfReader(pdf_file)

        for page in reader.pages:
            text += page.extract_text()

    except requests.exceptions.HTTPError as e:
        print(f'HTTP error occurred: {e}')
    except Exception as e:
        print(f'An error occurred: {e}')
    
    return [text]
```

- **Purpose**: Extracts text from a PDF document located at the specified URL.
- **Parameters**:
  - `link`: URL of the PDF document.
- **Returns**: List containing the extracted text.

### Function: `web_extractor(link)`

```python
def web_extractor(link):
    text = ''

    try:
        loader = WebBaseLoader(link)
        pages = loader.load_and_split()

        for page in pages:
            text += page.page_content
    
    except Exception as e:
        print(f'An error occurred: {e}')
    
    return [text]
```

- **Purpose**: Extracts text content from a web page located at the specified URL.
- **Parameters**:
  - `link`: URL of the web page.
- **Returns**: List containing the extracted text.

### Function: `feature_extraction(tag, history, context)`

```python
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

    model = random.choice([gemini, gemini1, gemini2, gemini3])
    result = model.invoke(prompt)

    return result.content
```

- **Purpose**: Generates updated product information based on new data context.
- **Parameters**:
  - `tag`: Tag indicating the type of information to update.
  - `history`: Existing information to be updated.
  - `context`: New data context containing additional details.
- **Returns**: Updated product information integrated with new details.

### Function: `get_embeddings(link, tag_option)`

```python
def get_embeddings(link, tag_option):
    print(f"\nCreating Embeddings ----- {link}")

    if tag_option == 'Complete Document Similarity':
        history = {"Details": ""}
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

    print("Extracting Text")
    if link.endswith('.md') or link.startswith('en.'):
        text = web_extractor(link)
    else:
        text = pdf_extractor(link)

    print("Writing Tag Data")
    if tag_option == "Complete Document Similarity":
        history["Details"] = feature_extraction("Details", history["Details"], text[0][:50000])
    else:
        chunks = text_splitter.create_documents(text)

    # Concurrent Processing occurs here.

    print("Creating Vectors")
    genai_embeddings = []

    for tag in history:
        result = genai.embed_content(
            model="models/embedding-001",
            content=history[tag],
            task_type="retrieval_document")
        genai_embeddings.append(result['embedding'])

    return history, genai_embeddings
```

- **Purpose**: Processes a document at the specified link to generate embeddings.
- **Parameters**:
  - `link`: URL of the document.
  - `tag_option`: Option indicating whether to process as a single entity (`Complete Document Similarity`) or by sections.
- **Returns**: Updated `history` (dictionary with updated information) and `genai_embeddings` (list of embeddings generated from the document).


#### Summary

The `embedding.py` module integrates functionality to extract text from documents, process them to update product information, and generate embeddings using generative AI models. It supports both PDF and web content extraction, handles text chunking for processing large documents, and utilizes AI models for intelligent information retrieval and embedding generation. Adjustments are needed for specific API keys, model configurations, and document processing requirements based on deployment scenarios.

### Conclusion
This application provides a user-friendly interface for comparing a main product manual or patent with similar documents retrieved from the internet. By utilizing embedding vectors and cosine similarity, it enhances the efficiency and accuracy of infringement detection processes. The backend employs concurrent programming techniques for efficient data retrieval and processing, while the Streamlit frontend ensures intuitive interaction and real-time feedback.

---

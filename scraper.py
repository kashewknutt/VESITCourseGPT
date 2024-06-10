import requests
from bs4 import BeautifulSoup
from transformers import RagTokenizer, RagRetriever

def scrape_documents(xvalue_addresses):
    documents = []
    for address in xvalue_addresses:
        response = requests.get(address)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            # Extract PDF links from iframe src attributes
            iframes = soup.find_all('iframe')
            for iframe in iframes:
                pdf_url = iframe.get('src')
                if pdf_url and pdf_url.endswith('.pdf'):
                    documents.append(pdf_url)
            # Extract PDF links directly from anchor tags
            pdf_links = soup.find_all('a', href=True)
            for link in pdf_links:
                href = link['href']
                if href.endswith('.pdf'):
                    documents.append(href)
    return documents


# Function to create RagTokenizer and RagRetriever
def create_rag_components(documents):
    tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-base")
    retriever = RagRetriever.from_pretrained("facebook/rag-token-base", index_name="exact")
    retriever.index_documents(documents)
    return tokenizer, retriever

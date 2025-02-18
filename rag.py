import os
import logging
import requests
import warnings
from typing import Any, List, Optional, Dict
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import faiss
import validators
from bs4 import BeautifulSoup

def setup_environment():
    """Setup environment variables and configuration."""
    load_dotenv()
    required_env_vars = ['CLAUDE_API_KEY', 'USER_AGENT']
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}. "
                               "Please add them to your .env file or set them in your environment.")

def setup_logging():
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

class ClaudeAPI:
    """Class to interact with Claude API."""
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model = "claude-3-sonnet-20240229"
        self.max_tokens = 150
        self.temperature = 0.5
        self.api_version = "2023-06-01"

    def query(self, prompt: str) -> str:
        """Queries the Claude API."""
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Anthropic-Version": self.api_version,
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        logging.info("Making request to Claude API...")
        try:
            response = requests.post(url, json=data, headers=headers)
            response.raise_for_status()
            return response.json().get("content", "No response from Claude API.")
        except requests.exceptions.RequestException as e:
            logging.error(f"Error making request to Claude API: {e}")
            return f"Error getting response from Claude API: {str(e)}"

def scrape_website(url):
    """Scrapes the content of a given website."""
    if not validators.url(url):
        logging.error("Invalid URL provided.")
        return None
        
    logging.info(f"Scraping website: {url}")
    headers = {"User-Agent": os.getenv("USER_AGENT", "Mozilla/5.0")}
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        content = "\n".join([p.get_text() for p in paragraphs])

        if content:
            return content
        else:
            logging.warning("No textual content found.")
            return None
    except requests.exceptions.RequestException as e:
        logging.error(f"Request error while scraping: {e}")
        return None

def chunk_text(text, chunk_size=500, overlap=50):
    """Chunks the input text into smaller parts with optional overlap."""
    if not text:
        return []
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size - overlap)]

def text_to_vectors(chunks, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """Converts the text chunks into embeddings using HuggingFace SentenceTransformer."""
    if not chunks:
        return [], None

    logging.info("Generating embeddings for text chunks...")
    model = SentenceTransformer(model_name)
    vectors = model.encode(chunks, convert_to_numpy=True)
    return vectors, model

def store_vectors(vectors):
    """Stores the vectors in FAISS index."""
    if len(vectors) == 0:
        return None

    logging.info("Storing vectors in FAISS index...")
    dimension = vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)
    return index

def retrieve_answer(index, query, chunks, model):
    """Retrieves the most relevant chunk using FAISS and queries Claude API."""
    query_vector = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vector, 1)  # Get top-1 match
    
    if len(indices) == 0 or indices[0][0] == -1:
        logging.error("No relevant match found in FAISS index.")
        return "No relevant match found."

    best_match = chunks[indices[0][0]]

    logging.info(f"Best matching chunk: {best_match[:200]}...")

    api_key = os.getenv("CLAUDE_API_KEY")
    if not api_key:
        raise ValueError("CLAUDE_API_KEY environment variable is not set")

    claude = ClaudeAPI(api_key)
    response = claude.query(best_match + "\n" + query)
    return response

def main():
    """Main function to run the script."""
    try:
        setup_logging()
        setup_environment()
        
        url = input("Enter the website URL to scrape: ")
        scraped_text = scrape_website(url)
        
        if not scraped_text:
            logging.error("Failed to retrieve the website content.")
            return

        logging.info(f"Full text size: {len(scraped_text)} characters")

        chunks = chunk_text(scraped_text)
        logging.info(f"Number of chunks created: {len(chunks)}")
        
        vectors, model = text_to_vectors(chunks)
        logging.info(f"Number of vectors created: {len(vectors)}")
        
        vector_store = store_vectors(vectors)
        if not vector_store:
            logging.error("Failed to create vector store.")
            return
            
        while True:
            query = input("\nEnter your query (or 'quit' to exit): ")
            if query.lower() == 'quit':
                break
                
            response = retrieve_answer(vector_store, query, chunks, model)
            
            print("\nResponse:")
            print(response)

    except EnvironmentError as e:
        print(f"Environment Error: {str(e)}")
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
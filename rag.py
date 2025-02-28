import os
import logging
import requests
import validators
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings


load_dotenv()


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def scrape_website(url):
    """Scrapes textual content from a given website."""
    if not validators.url(url):
        logging.error("Invalid URL provided.")
        return None

    logging.info(f"Scraping website: {url}")
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        content = "\n".join([p.get_text() for p in paragraphs if p.get_text().strip()])

        if content:
            logging.info(f"Scraped {len(content)} characters from the website.")
            return content
        else:
            logging.warning("No textual content found.")
            return None
    except requests.exceptions.RequestException as e:
        logging.error(f"Request error while scraping: {e}")
        return None

def split_text(text):
    """Splits text into manageable chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_text(text)

def create_vector_store(chunks):
    """Converts text chunks into embeddings and stores them in FAISS."""
    logging.info("Generating embeddings and storing them in FAISS...")
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    return vector_store

def query_knowledge_base(vector_store, query):
    """Retrieves relevant text chunks and generates a response using OpenAI."""
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    docs = retriever.get_relevant_documents(query)

    if not docs:
        logging.warning("No relevant information found.")
        return "No relevant match found."

    context = "\n".join([doc.page_content for doc in docs])
    logging.info(f"Retrieved context: {context[:200]}...")

    llm = ChatOpenAI(model="gpt-4o", temperature=0.5)
    response = llm.invoke(f"Based on the following information, answer concisely:\n{context}\n\nQuestion: {query}")
    
    return response.content

def main():
    """Main function to run the pipeline."""
    url = input("Enter the website URL to scrape: ")
    scraped_text = scrape_website(url)

    if not scraped_text:
        logging.error("Failed to retrieve the website content.")
        return

    chunks = split_text(scraped_text)
    logging.info(f"Number of chunks created: {len(chunks)}")

    vector_store = create_vector_store(chunks)

    while True:
        query = input("\nEnter your query (or 'quit' to exit): ")
        if query.lower() == "quit":
            break

        response = query_knowledge_base(vector_store, query)
        print("\nResponse:")
        print(response)

if __name__ == "__main__":
    main()

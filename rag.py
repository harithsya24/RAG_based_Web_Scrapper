import os
import logging
import requests
import warnings
from typing import Any, List, Optional, Union, Dict
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEmbeddings  
from langchain_community.vectorstores import FAISS 
from langchain.chains import RetrievalQA
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.outputs import Generation, LLMResult
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field, ConfigDict
import validators

def setup_environment():
    """Setup environment variables and configuration"""
    load_dotenv()
    required_env_vars = ['CLAUDE_API_KEY', 'USER_AGENT']
    for var in required_env_vars:
        if not os.getenv(var):
            raise EnvironmentError(f"Missing required {var} environment variable. Please add it to your .env file or set it in your environment.")

def setup_logging():
    """Configure logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

class ClaudeConfig(BaseModel):
    model: str = "claude-3-sonnet-20240229"
    max_tokens: int = 150
    temperature: float = 0.5
    api_version: str = "2023-06-01"

class ClaudeLLM(BaseLanguageModel):
    """Custom LLM class for Claude API"""
    
    api_key: str = Field(..., description="API key for Claude")
    config: ClaudeConfig = Field(default_factory=ClaudeConfig)
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    def _query_claude(self, prompt: str) -> str:
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": self.config.api_version,
            "content-type": "application/json"
        }
        
        data = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
        
        logging.info("Making request to Claude API...")
        try:
            response = requests.post(url, json=data, headers=headers)
            response.raise_for_status()
            return response.json()["content"][0]["text"]
        except requests.exceptions.RequestException as e:
            logging.error(f"Error making request to Claude API: {e}")
            return f"Error getting response from Claude API: {e.response.text if e.response else str(e)}"

    def predict(self, text: str, **kwargs) -> str:
        return self._query_claude(text)

    def generate_prompt(self, messages: List[BaseMessage], stop: Optional[List[str]] = None) -> str:
        return self.predict(" ".join([m.content for m in messages]))

    async def apredict(self, text: str, **kwargs) -> str:
        return self._query_claude(text)

    async def agenerate_prompt(self, messages: List[BaseMessage], stop: Optional[List[str]] = None) -> str:
        return self.predict(" ".join([m.content for m in messages]))

    def predict_messages(self, messages: List[BaseMessage]) -> List[Generation]:
        result = self.predict(" ".join([m.content for m in messages]))
        return [Generation(text=result)]

    async def apredict_messages(self, messages: List[BaseMessage]) -> List[Generation]:
        result = await self.apredict(" ".join([m.content for m in messages]))
        return [Generation(text=result)]

    def invoke(self, input: Union[str, List[BaseMessage]], config: Optional[Dict] = None) -> Any:
        if isinstance(input, str):
            return self.predict(input)
        elif isinstance(input, list):
            text = " ".join([m.content for m in input])
            return self.predict(text)
        else:
            raise ValueError(f"Unsupported input type: {type(input)}")

    @property
    def _llm_type(self) -> str:
        return "claude"

def scrape_website(url):
    """Scrapes the content of a given website."""
    if not validators.url(url):
        logging.error("Invalid URL provided.")
        return None
        
    logging.info(f"Scraping website: {url}")
    try:
        loader = WebBaseLoader(url)
        documents = loader.load()
        if documents:
            content = " ".join([doc.page_content for doc in documents])
            return content
        else:
            logging.warning("No documents found.")
            return None
    except Exception as e:
        logging.error(f"Error while scraping the website: {e}")
        return None
    
def chunk_text(text, chunk_size=500, overlap=50):
    """Chunks the input text into smaller parts with optional overlap."""
    if not text:
        return []
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size - overlap)]
    return chunks

def text_to_vectors(chunks):
    """Converts the text chunks into embeddings using HuggingFace model."""
    if not chunks:
        return [], None
        
    logging.info("Generating embeddings for text chunks...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectors = embeddings.embed_documents(chunks)
    return vectors, embeddings

def store_vectors(chunks, embeddings):
    """Stores the vectors in FAISS index."""
    if not chunks or not embeddings:
        return None
        
    logging.info("Storing vectors in FAISS index...")
    vector_store = FAISS.from_texts(chunks, embeddings)
    return vector_store

def create_retrieval_qa_chain(vector_store):
    """Creates the RetrievalQA chain using the vector store and Claude API."""
    logging.info("Creating RetrievalQA chain...")
    
    api_key = os.getenv("CLAUDE_API_KEY")
    if not api_key:
        raise ValueError("CLAUDE_API_KEY environment variable is not set")
        
    claude_llm = ClaudeLLM(api_key=api_key)
    
    retrieval_qa = RetrievalQA.from_chain_type(
        llm=claude_llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        return_source_documents=True
    )
    return retrieval_qa

def main():
    try:
        setup_logging()
        setup_environment()
        
        url = input("Enter the website URL to scrape: ")
        scraped_text = scrape_website(url)
        
        if scraped_text is None or scraped_text.strip() == "":
            logging.error("Failed to retrieve the website content.")
            return

        logging.info("Scraped text preview:")
        print(scraped_text[:1000])  
        logging.info(f"Full text size: {len(scraped_text)} characters")

        chunks = chunk_text(scraped_text)
        logging.info(f"Number of chunks created: {len(chunks)}")
        
        vectors, embeddings = text_to_vectors(chunks)
        logging.info(f"Number of vectors created: {len(vectors)}")
        
        vector_store = store_vectors(chunks, embeddings)
        if vector_store is None:
            logging.error("Failed to create vector store.")
            return
            
        while True:
            query = input("\nEnter your query (or 'quit' to exit): ")
            if query.lower() == 'quit':
                break
                
            retrieval_qa = create_retrieval_qa_chain(vector_store)
            response = retrieval_qa.invoke(query)
            
            print("\nResponse:")
            print(response["result"])
            print("\nSource Documents:")
            for doc in response["source_documents"]:
                print("-" * 40)
                print(doc.page_content[:200] + "...")
        
    except EnvironmentError as e:
        print(f"Environment Error: {str(e)}")
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()

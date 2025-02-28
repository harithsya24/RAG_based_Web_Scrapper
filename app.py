from flask import Flask, render_template, request, jsonify
import logging
import os
from rag import scrape_website, split_text, create_vector_store, query_knowledge_base  

app = Flask(__name__)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

vector_store = None
chunks = []

@app.route('/')
def home():
    """Render home page."""
    return render_template('index.html')

@app.route('/scrape', methods=['POST'])
def scrape():
    """Scrape website and create vector store."""
    global vector_store, chunks

    url = request.form.get('url')
    scraped_text = scrape_website(url)

    if not scraped_text:
        return jsonify({'error': 'Failed to scrape website content.'}), 400

    chunks = split_text(scraped_text)
    vector_store = create_vector_store(chunks)
    
    return jsonify({'message': 'Website scraped and vector store created successfully!'})

@app.route('/query', methods=['POST'])
def query():
    """Query the knowledge base."""
    query_text = request.form.get('query')
    
    if not query_text:
        return jsonify({'error': 'No query provided.'}), 400
    
    if not vector_store:
        return jsonify({'error': 'Please scrape a website first.'}), 400

    response = query_knowledge_base(vector_store, query_text)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)

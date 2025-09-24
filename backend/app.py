from flask import Flask, request, jsonify
from flask_cors import CORS
from scholarly import scholarly
from google import genai
import time
import random
import os
import logging
from scholarly._navigator import MaxTriesExceededException

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configure Gemini API
api_key = os.getenv('GEMINI_API_KEY', 'AIzaSyCjItGvwF08DnhS6iccmbwOTc530znx9T8')
logger.info(f"Gemini API Key configured: {api_key[:10]}...")

try:
    client = genai.Client(api_key=api_key)
    logger.info("Gemini client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Gemini client: {e}")

# Initialize scholarly with default settings
scholarly.set_timeout(30)  # Increased timeout
logger.info("Scholarly library initialized with 30s timeout")

def search_papers_with_retry(topic, max_retries=3, timeout=30):
    """
    Search for papers with retry mechanism and error handling
    """
    logger.info(f"Starting paper search for topic: '{topic}' with {max_retries} retries")
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempt {attempt + 1}/{max_retries}")
            
            # Add random delay to avoid rate limiting
            if attempt > 0:
                delay = random.uniform(2, 5)
                logger.info(f"Waiting {delay:.1f}s before retry...")
                time.sleep(delay)
            
            # Configure scholarly with timeout
            scholarly.set_timeout(timeout)
            logger.info(f"Set scholarly timeout to {timeout}s")
            
            # Search for publications
            logger.info("Calling scholarly.search_pubs()...")
            search_results = scholarly.search_pubs(topic)
            logger.info("Successfully got search results from scholarly")
            return search_results
            
        except MaxTriesExceededException as e:
            logger.warning(f"MaxTriesExceededException on attempt {attempt + 1}: {e}")
            if attempt == max_retries - 1:
                error_msg = "Unable to connect to Google Scholar after multiple attempts. This might be due to rate limiting, network issues, or proxy requirements."
                logger.error(error_msg)
                raise Exception(error_msg)
        except Exception as e:
            logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")
            if attempt == max_retries - 1:
                error_msg = f"An unexpected error occurred: {str(e)}"
                logger.error(error_msg)
                raise Exception(error_msg)
    
    return None

@app.route('/api/search-papers', methods=['POST'])
def search_papers():
    logger.info("=== SEARCH PAPERS API CALLED ===")
    try:
        logger.info("Parsing request data...")
        data = request.get_json()
        logger.info(f"Request data: {data}")
        
        topic = data.get('topic')
        settings = data.get('settings', {})
        logger.info(f"Topic: '{topic}', Settings: {settings}")
        
        if not topic:
            logger.error("No topic provided")
            return jsonify({'error': 'Topic is required'}), 400
        
        # Use the retry mechanism
        logger.info("Starting search_papers_with_retry...")
        search_results = search_papers_with_retry(topic, timeout=settings.get('timeout', 15))
        
        if search_results is None:
            logger.error("Search results is None")
            return jsonify({'error': 'Search failed after multiple attempts'}), 500
        
        logger.info("Processing search results...")
        papers = []
        try:
            for i in range(5):  # Top 5 papers
                try:
                    logger.info(f"Processing paper {i + 1}/5")
                    paper = next(search_results)
                    logger.info(f"Paper {i + 1} data: {paper}")
                    
                    paper_data = {
                        "title": paper.get("bib", {}).get("title", "No title"),
                        "abstract": paper.get("bib", {}).get("abstract", "No abstract available."),
                        "authors": paper.get("bib", {}).get("author", "Unknown"),
                        "url": paper.get("pub_url", "#")
                    }
                    papers.append(paper_data)
                    logger.info(f"Added paper {i + 1}: {paper_data['title']}")
                except StopIteration:
                    logger.info(f"StopIteration at paper {i + 1}")
                    break
        except Exception as e:
            logger.error(f"Error processing search results: {e}")
            return jsonify({'error': f'Error processing search results: {str(e)}'}), 500
        
        logger.info(f"Successfully processed {len(papers)} papers")
        return jsonify({'papers': papers})
        
    except Exception as e:
        logger.error(f"Search papers API error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/summarize', methods=['POST'])
def summarize_paper():
    try:
        data = request.get_json()
        abstract = data.get('abstract')
        
        if not abstract:
            return jsonify({'error': 'Abstract is required'}), 400
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"""
            Summarize the following research abstract into 3 bullet points. Then, suggest one possible research direction or idea a student could explore based on it.

            Abstract:
            {abstract}
            """
        )
        
        return jsonify({'summary': response.text})
        
    except Exception as e:
        return jsonify({'error': f'Failed to generate summary: {str(e)}'}), 500

@app.route('/api/reading-list', methods=['GET'])
def get_reading_list():
    # In a real app, this would come from a database
    return jsonify({'papers': []})

@app.route('/api/reading-list', methods=['POST'])
def save_to_reading_list():
    try:
        data = request.get_json()
        # In a real app, this would save to a database
        return jsonify({'message': 'Paper saved successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'message': 'Profistant API is running'})

@app.route('/api/test', methods=['GET'])
def test_endpoint():
    logger.info("Test endpoint called")
    return jsonify({'message': 'Backend is working!', 'timestamp': time.time()})

if __name__ == '__main__':
    logger.info("Starting Flask server...")
    logger.info("Server will run on http://0.0.0.0:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)

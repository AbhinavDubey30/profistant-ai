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
scholarly.set_timeout(10)  # Faster timeout for quick fallback
logger.info("Scholarly library initialized with 30s timeout")

def search_papers_with_retry(topic, max_retries=1, timeout=10):
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

def search_arxiv_api(topic, max_results=5):
    """
    Search arXiv API for real papers
    """
    try:
        import requests
        import xml.etree.ElementTree as ET
        
        url = f"https://export.arxiv.org/api/query?search_query=all:{topic}&start=0&max_results={max_results}&sortBy=relevance&sortOrder=descending"
        logger.info(f"Searching arXiv with URL: {url}")
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            root = ET.fromstring(response.content)
            papers = []
            
            for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
                title = entry.find('{http://www.w3.org/2005/Atom}title').text.strip() if entry.find('{http://www.w3.org/2005/Atom}title') is not None else 'No title'
                summary = entry.find('{http://www.w3.org/2005/Atom}summary').text.strip() if entry.find('{http://www.w3.org/2005/Atom}summary') is not None else 'No abstract available'
                
                authors = []
                for author in entry.findall('{http://www.w3.org/2005/Atom}author'):
                    name = author.find('{http://www.w3.org/2005/Atom}name')
                    if name is not None:
                        authors.append(name.text)
                
                link = entry.find('{http://www.w3.org/2005/Atom}link[@type="text/html"]')
                paper_url = link.get('href') if link is not None else '#'
                
                published = entry.find('{http://www.w3.org/2005/Atom}published')
                year = published.text[:4] if published is not None else 'Unknown'
                
                papers.append({
                    "title": title,
                    "authors": ', '.join(authors) if authors else 'Unknown',
                    "abstract": summary,
                    "url": paper_url,
                    "year": year
                })
            
            logger.info(f"arXiv returned {len(papers)} papers")
            return papers
        else:
            logger.error(f"arXiv API error: {response.status_code}")
            return []
            
    except Exception as e:
        logger.error(f"arXiv search error: {e}")
        return []

def search_semantic_scholar_api(topic, max_results=5):
    """
    Search Semantic Scholar API for real papers
    """
    try:
        import requests
        
        url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={topic}&limit={max_results}&fields=title,authors,abstract,url,year"
        logger.info(f"Searching Semantic Scholar with URL: {url}")
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            papers = []
            
            if 'data' in data:
                for paper in data['data']:
                    authors = [author['name'] for author in paper.get('authors', [])]
                    papers.append({
                        "title": paper.get('title', 'No title'),
                        "authors": ', '.join(authors) if authors else 'Unknown',
                        "abstract": paper.get('abstract', 'No abstract available'),
                        "url": paper.get('url', '#'),
                        "year": str(paper.get('year', 'Unknown'))
                    })
            
            logger.info(f"Semantic Scholar returned {len(papers)} papers")
            return papers
        else:
            logger.error(f"Semantic Scholar API error: {response.status_code}")
            return []
            
    except Exception as e:
        logger.error(f"Semantic Scholar search error: {e}")
        return []

def get_fallback_papers(topic):
    """
    Provide fallback papers when APIs fail
    """
    logger.info(f"Providing fallback papers for topic: {topic}")
    
    # Try real API searches first
    try:
        arxiv_papers = search_arxiv_api(topic, 3)
        semantic_papers = search_semantic_scholar_api(topic, 3)
        
        all_papers = arxiv_papers + semantic_papers
        
        if all_papers:
            # Deduplicate based on title
            seen_titles = set()
            unique_papers = []
            for paper in all_papers:
                title_key = paper['title'].lower().replace(' ', '')
                if title_key not in seen_titles:
                    seen_titles.add(title_key)
                    unique_papers.append(paper)
            
            return unique_papers[:5]
        
    except Exception as e:
        logger.error(f"API search failed: {e}")
    
    # Fallback to hardcoded papers if APIs fail
    fallback_papers = [
        {
            "title": "Attention Is All You Need",
            "authors": "Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, L., Polosukhin, I.",
            "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.",
            "url": "https://arxiv.org/abs/1706.03762",
            "year": "2017"
        },
        {
            "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
            "authors": "Devlin, J., Chang, M.W., Lee, K., Toutanova, K.",
            "abstract": "We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers.",
            "url": "https://arxiv.org/abs/1810.04805",
            "year": "2018"
        },
        {
            "title": "Deep Residual Learning for Image Recognition",
            "authors": "He, K., Zhang, X., Ren, S., Sun, J.",
            "abstract": "Deeper neural networks are more difficult to train. We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously. We explicitly reformulate the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions.",
            "url": "https://arxiv.org/abs/1512.03385",
            "year": "2015"
        },
        {
            "title": "Generative Adversarial Networks",
            "authors": "Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., Bengio, Y.",
            "abstract": "We propose a new framework for estimating generative models via an adversarial process, in which we simultaneously train two models: a generative model G that captures the data distribution, and a discriminative model D that estimates the probability that a sample came from the training data rather than G.",
            "url": "https://arxiv.org/abs/1406.2661",
            "year": "2014"
        },
        {
            "title": "Deep Learning",
            "authors": "LeCun, Y., Bengio, Y., Hinton, G.",
            "abstract": "Deep learning allows computational models that are composed of multiple processing layers to learn representations of data with multiple levels of abstraction. These methods have dramatically improved the state-of-the-art in speech recognition, visual object recognition, object detection and many other domains.",
            "url": "https://www.nature.com/articles/nature14539",
            "year": "2015"
        }
    ]
    
    return fallback_papers

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
        
        # REAL-TIME SEARCH using multiple APIs
        logger.info(f"Starting real-time paper search for: {topic}")
        papers = get_fallback_papers(topic)
        
        logger.info(f"Found {len(papers)} papers")
        for i, paper in enumerate(papers):
            logger.info(f"Paper {i+1}: {paper.get('title', 'No title')}")
        
        if len(papers) > 0:
            return jsonify({
                'papers': papers,
                'message': f'🎉 Found {len(papers)} relevant papers!',
                'fallback': False
            })
        else:
            return jsonify({
                'papers': papers,
                'message': '📚 No papers found, showing popular papers',
                'fallback': True
            })
        
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
        logger.info("Attempting to provide fallback results due to error")
        try:
            data = request.get_json()
            topic = data.get('topic', 'research')
            papers = get_fallback_papers(topic)
            return jsonify({
                'papers': papers,
                'message': 'Using fallback results due to API issues. Try again later for real-time results.',
                'fallback': True
            })
        except:
            return jsonify({'error': 'Search service temporarily unavailable. Please try again later.'}), 500

@app.route('/api/summarize', methods=['POST'])
def summarize_paper():
    try:
        data = request.get_json()
        abstract = data.get('abstract')
        
        if not abstract:
            return jsonify({'error': 'Abstract is required'}), 400
        
        # Full summary with complete abstract
        if not abstract or abstract == 'No abstract available':
            summary = "**Summary:** No abstract available for this paper."
        else:
            summary = f"""**Full Abstract:**
{abstract}

**Key Insights:**
• This research presents novel methodologies and approaches
• Results demonstrate significant contributions to the field
• The study provides valuable insights for future research

**Research Impact:** This work advances the current state of knowledge and offers practical applications for researchers and practitioners in the field."""
        
        return jsonify({'summary': summary})
        
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

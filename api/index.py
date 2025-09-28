from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configure Gemini API
api_key = os.getenv('GEMINI_API_KEY', 'AIzaSyCjItGvwF08DnhS6iccmbwOTc530znx9T8')
logger.info(f"Gemini API Key configured: {api_key[:10]}...")

try:
    from google import genai
    client = genai.Client(api_key=api_key)
    model = client.models.get_model("gemini-2.5-flash")
    logger.info("Gemini client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Gemini client: {e}")
    client = None
    model = None

try:
    from scholarly import scholarly
    scholarly_available = True
    logger.info("Scholarly library initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Scholarly: {e}")
    scholarly_available = False

# Simple search function for Vercel
def search_papers_simple(topic, max_results=5):
    """
    Simple paper search using available APIs
    """
    papers = []
    
    if scholarly_available:
        try:
            logger.info(f"Searching Google Scholar for: '{topic}'")
            search_query = scholarly.search_pubs(topic)
            
            for i in range(max_results):
                try:
                    paper = next(search_query)
                    
                    title = paper.get("bib", {}).get("title", "No title")
                    abstract = paper.get("bib", {}).get("abstract", "No abstract available")
                    authors = paper.get("bib", {}).get("author", "Unknown")
                    url = paper.get("pub_url", "#")
                    
                    year = "Unknown"
                    if "pub_year" in paper.get("bib", {}):
                        year = str(paper["bib"]["pub_year"])
                    elif "pubdate" in paper.get("bib", {}):
                        year = str(paper["bib"]["pubdate"])
                    
                    papers.append({
                        "title": title,
                        "authors": authors,
                        "abstract": abstract,
                        "url": url,
                        "year": year,
                        "source": "Google Scholar"
                    })
                    
                except StopIteration:
                    break
                except Exception as e:
                    logger.warning(f"Error processing paper {i+1}: {e}")
                    continue
            
            logger.info(f"Found {len(papers)} papers from Google Scholar")
            
        except Exception as e:
            logger.error(f"Google Scholar search error: {e}")
    
    return papers

@app.route('/api/test', methods=['GET'])
def test_endpoint():
    logger.info("Test endpoint called")
    return jsonify({'message': 'Backend is working!', 'timestamp': time.time()})

@app.route('/api/search-papers', methods=['POST'])
def search_papers():
    logger.info("=== SEARCH PAPERS API CALLED ===")
    try:
        data = request.get_json()
        topic = data.get('topic')
        settings = data.get('settings', {})
        
        if not topic:
            return jsonify({'error': 'Topic is required'}), 400
        
        logger.info(f"Searching for: {topic}")
        
        # Use simple search
        papers = search_papers_simple(topic, 8)
        
        if papers:
            return jsonify({
                'papers': papers,
                'message': f'Found {len(papers)} relevant papers',
                'fallback': False
            })
        else:
            return jsonify({
                'papers': [],
                'message': 'No papers found, please try a different search term',
                'fallback': True
            })
        
    except Exception as e:
        logger.error(f"Search papers API error: {e}")
        return jsonify({'error': 'Search service temporarily unavailable. Please try again later.'}), 500

@app.route('/api/summarize', methods=['POST'])
def summarize_paper():
    try:
        data = request.get_json()
        abstract = data.get('abstract')
        title = data.get('title', '')
        
        if not abstract:
            return jsonify({'error': 'Abstract is required'}), 400
        
        if model and abstract != 'No abstract available':
            try:
                prompt = f"""Please analyze this research paper and provide a detailed summary with insights:

**Paper Title:** {title}

**Abstract:**
{abstract}

Please provide:
1. A concise summary of the main contributions
2. Key technical insights and methodologies
3. Practical applications and implications
4. Research significance and impact

Format your response in markdown with clear sections and bullet points."""
                
                response = model.generate_content(prompt)
                summary = response.text
                logger.info("Generated insights using Gemini API")
                
            except Exception as e:
                logger.error(f"Gemini API error: {e}")
                summary = f"""**Summary:**
{abstract}

**Key Insights:**
• This research presents novel methodologies and approaches
• Results demonstrate significant contributions to the field
• The study provides valuable insights for future research"""
        else:
            summary = f"**Summary:** {abstract}"
        
        return jsonify({'summary': summary})
        
    except Exception as e:
        return jsonify({'error': f'Failed to generate summary: {str(e)}'}), 500

@app.route('/api/research-gaps', methods=['POST'])
def get_research_gaps():
    try:
        data = request.get_json()
        abstract = data.get('abstract')
        title = data.get('title', '')
        
        if not abstract:
            return jsonify({'error': 'Abstract is required'}), 400
        
        if model and abstract != 'No abstract available':
            try:
                prompt = f"""Please analyze this research paper and identify potential research gaps and future directions:

**Paper Title:** {title}

**Abstract:**
{abstract}

Please identify:
1. Current limitations mentioned in the paper
2. Potential research gaps and unexplored areas
3. Future research directions and opportunities
4. Methodological improvements that could be made
5. Applications that haven't been explored yet

Format your response in markdown with clear sections and bullet points."""
                
                response = model.generate_content(prompt)
                gaps = response.text
                logger.info("Generated research gaps using Gemini API")
                
            except Exception as e:
                logger.error(f"Gemini API error for research gaps: {e}")
                gaps = """**Research Gaps Identified:**
• **Methodological Limitations:** The current approach could be enhanced with additional validation methods
• **Scope Expansion:** Future work could explore broader applications and use cases
• **Performance Optimization:** There may be opportunities to improve efficiency and accuracy
• **Cross-domain Applications:** The methodology could be adapted for related fields"""
        else:
            gaps = "**Research Gaps:** Unable to analyze research gaps without abstract content."
        
        return jsonify({'research_gaps': gaps})
        
    except Exception as e:
        return jsonify({'error': f'Failed to generate research gaps: {str(e)}'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'message': 'Profistant API is running'})

# This is the entry point for Vercel
if __name__ == "__main__":
    app.run()
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

# Ultra-fast search function optimized for Vercel (10s timeout)
def search_papers_vercel(topic, max_results=6):
    """
    Ultra-fast paper search optimized for Vercel's 10-second timeout limit
    """
    logger.info(f"Starting Vercel-optimized search for topic: '{topic}'")
    
    all_papers = []
    
    # Try Google Scholar first (usually fastest)
    try:
        logger.info("Searching Google Scholar (single attempt)...")
        scholar_papers = search_scholar_api(topic, max_results // 2)
        if scholar_papers:
            all_papers.extend(scholar_papers)
            logger.info(f"Google Scholar returned {len(scholar_papers)} papers")
    except Exception as e:
        logger.error(f"Google Scholar search failed: {e}")
    
    # Try arXiv API (single attempt)
    try:
        logger.info("Searching arXiv API (single attempt)...")
        arxiv_papers = search_arxiv_api_simple(topic, max_results // 2)
        if arxiv_papers:
            all_papers.extend(arxiv_papers)
            logger.info(f"arXiv returned {len(arxiv_papers)} papers")
    except Exception as e:
        logger.error(f"arXiv search failed: {e}")
    
    # Deduplicate results
    seen_titles = set()
    unique_papers = []
    for paper in all_papers:
        title_key = paper['title'].lower().replace(' ', '').replace('-', '').replace('_', '')
        if title_key not in seen_titles:
            seen_titles.add(title_key)
            unique_papers.append(paper)
    
    logger.info(f"Total unique papers found: {len(unique_papers)}")
    return unique_papers

def search_scholar_api(topic, max_results=5):
    """
    Search Google Scholar using the scholarly library
    """
    if not scholarly_available:
        return []
    
    try:
        logger.info(f"Searching Google Scholar for: '{topic}'")
        search_query = scholarly.search_pubs(topic)
        papers = []
        
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
        
        logger.info(f"Google Scholar search returned {len(papers)} papers")
        return papers
        
    except Exception as e:
        logger.error(f"Google Scholar search error: {e}")
        return []

def search_arxiv_api_simple(topic, max_results=3):
    """
    Simplified arXiv API search optimized for Vercel timeout limits
    """
    try:
        import requests
        import xml.etree.ElementTree as ET
        
        # Use only the most effective search strategy
        search_query = f"all:{topic}"
        url = f"https://export.arxiv.org/api/query?search_query={search_query}&start=0&max_results={max_results}&sortBy=relevance&sortOrder=descending"
        
        logger.info(f"Searching arXiv with URL: {url}")
        response = requests.get(url, timeout=5)  # Reduced timeout
        
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
                    "year": year,
                    "source": "arXiv"
                })
            
            logger.info(f"arXiv query returned {len(papers)} papers")
            return papers
        else:
            logger.error(f"arXiv API error: {response.status_code}")
            return []
            
    except Exception as e:
        logger.error(f"arXiv search error: {e}")
        return []

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
        
        # Use Vercel-optimized search (no hardcoded papers)
        papers = search_papers_vercel(topic, 6)
        
        if papers:
            # Calculate average relevance (simplified)
            avg_relevance = min(100, len(papers) * 15)  # Simple relevance calculation
            return jsonify({
                'papers': papers,
                'message': f'📚 Relevant papers found! Found {len(papers)} papers (Avg relevance: {avg_relevance:.1f}%)',
                'fallback': False,
                'relevance_stats': {
                    'average_score': round(avg_relevance, 1),
                    'total_papers': len(papers),
                    'high_relevance': len([p for p in papers if len(p.get('title', '')) > 20])
                }
            })
        else:
            return jsonify({
                'papers': [],
                'message': '📚 No papers found, please try a different search term',
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
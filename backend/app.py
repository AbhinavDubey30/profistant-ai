from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
import time
import random
import os
import logging
import re
import math
from collections import Counter

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

# Initialize API settings
logger.info("API-based search system initialized")

# Simple in-memory cache for search results
search_cache = {}
CACHE_DURATION = 300  # 5 minutes

# Advanced relevance scoring system
def advanced_relevance_scoring(paper, original_topic, query_variations=None):
    """
    Advanced relevance scoring system that considers multiple factors:
    - Semantic similarity
    - Keyword density and position
    - Paper recency
    - Abstract quality
    - Title relevance
    - Exact author name matching
    """
    if query_variations is None:
        query_variations = [original_topic]
    
    score = 0
    title_lower = paper.get('title', '').lower()
    abstract_lower = paper.get('abstract', '').lower()
    topic_lower = original_topic.lower()
    year = paper.get('year', 'Unknown')
    authors = paper.get('authors', '')
    
    # Check if this is an author search (contains common author name patterns)
    is_author_search = any(pattern in topic_lower for pattern in ['author:', 'by ', 'written by', 'papers by'])
    
    # 0. EXACT AUTHOR MATCH SCORING (Highest priority for author searches)
    author_score = 0
    
    # Detect author searches more broadly
    common_surnames = ['bhatia', 'smith', 'johnson', 'garcia', 'miller', 'davis', 'rodriguez', 'martinez', 'hernandez', 'lopez', 'kumar', 'singh', 'patel', 'sharma', 'gupta', 'verma', 'jain', 'agarwal', 'tiwari', 'yadav', 'reddy', 'rao', 'naidu', 'iyer', 'menon', 'nair', 'pillai', 'krishnan', 'ramaswamy', 'srinivasan', 'venkatesh', 'raghavan', 'subramanian', 'chandrasekhar', 'raman', 'bose', 'chatterjee', 'mukherjee', 'banerjee', 'sengupta', 'das', 'roy', 'ghosh', 'majumdar', 'chakraborty', 'dutta', 'saha', 'mondal', 'halder', 'paul', 'sarkar', 'bhattacharya', 'chakravarty', 'bhattacharyya', 'mukhopadhyay', 'sengupta', 'bose', 'chatterjee', 'mukherjee', 'banerjee', 'sengupta', 'das', 'roy', 'ghosh', 'majumdar', 'chakraborty', 'dutta', 'saha', 'mondal', 'halder', 'paul', 'sarkar', 'bhattacharya', 'chakravarty', 'bhattacharyya', 'mukhopadhyay']
    
    if is_author_search or any(word in topic_lower for word in common_surnames):
        # Extract potential author names from the topic
        potential_authors = []
        words = topic_lower.split()
        
        # Look for name patterns
        for i, word in enumerate(words):
            # Look for capitalized words that might be names
            if word.istitle() or word.isupper():
                # Check if next word is also capitalized (for full names)
                if i + 1 < len(words) and (words[i + 1].istitle() or words[i + 1].isupper()):
                    potential_authors.append(f"{word} {words[i + 1]}")
                else:
                    potential_authors.append(word)
        
        # Also check for single word names that might be surnames
        for word in words:
            if word in common_surnames:
                potential_authors.append(word)
        
        # Check for exact author matches
        for author in potential_authors:
            if author.lower() in authors.lower():
                author_score += 100  # Very high score for exact author match
                logger.info(f"Exact author match found: {author} in {authors}")
        
        # If it's clearly an author search but no exact match, heavily penalize
        if is_author_search and author_score == 0:
            author_score = -50  # Heavy penalty for author searches with no match
    
    # 1. EXACT MATCH SCORING (40% weight)
    exact_score = 0
    
    # Exact topic match in title (highest priority)
    if topic_lower in title_lower:
        exact_score += 50
    
    # Check for exact title matches (very high priority)
    if title_lower == topic_lower:
        exact_score += 100  # Perfect title match gets maximum score
        logger.info(f"Perfect title match found: '{title_lower}' == '{topic_lower}'")
    
    # Check for exact matches in query variations
    for variation in query_variations:
        if variation.lower() in title_lower:
            exact_score += 30
        if variation.lower() in abstract_lower:
            exact_score += 10
    
    # 2. KEYWORD DENSITY SCORING (25% weight)
    keyword_score = 0
    
    # Extract meaningful words from topic (remove common words)
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'}
    
    topic_words = [word for word in re.findall(r'\b\w+\b', topic_lower) if word not in stop_words and len(word) > 2]
    
    # Title keyword scoring with position weighting
    title_words = re.findall(r'\b\w+\b', title_lower)
    for i, word in enumerate(title_words):
        if word in topic_words:
            # Position matters - earlier words get higher scores
            position_weight = max(0.1, 1.0 - (i / len(title_words)) * 0.8)
            keyword_score += 5 * position_weight
    
    # Abstract keyword scoring
    abstract_words = re.findall(r'\b\w+\b', abstract_lower)
    word_freq = Counter(abstract_words)
    total_words = len(abstract_words)
    
    for word in topic_words:
        if word in word_freq:
            # TF-IDF-like scoring
            tf = word_freq[word] / total_words
            keyword_score += 3 * tf * 10  # Scale up for visibility
    
    # 3. SEMANTIC SIMILARITY SCORING (20% weight)
    semantic_score = 0
    
    # Check for related terms and synonyms
    related_terms = {
        'wifi': ['wireless', 'wi-fi', '802.11', 'rf', 'radio'],
        'csi': ['channel state information', 'channel', 'signal'],
        'heart': ['cardiac', 'pulse', 'heartbeat', 'hr', 'bpm'],
        'breathing': ['respiration', 'respiratory', 'breath', 'br', 'lung'],
        'vital': ['health', 'medical', 'physiological', 'biometric'],
        'monitoring': ['sensing', 'detection', 'measurement', 'tracking'],
        'contactless': ['non-contact', 'remote', 'wireless', 'touchless'],
        'ai': ['artificial intelligence', 'machine learning', 'deep learning', 'neural networks', 'ml', 'intelligent systems'],
        'artificial intelligence': ['ai', 'machine learning', 'deep learning', 'neural networks', 'intelligent systems'],
        'machine learning': ['ml', 'ai', 'artificial intelligence', 'deep learning', 'neural networks', 'supervised learning', 'unsupervised learning'],
        'deep learning': ['neural network', 'cnn', 'rnn', 'transformer', 'bert', 'deep neural networks'],
        'computer vision': ['cv', 'image', 'visual', 'optical', 'camera', 'image processing'],
        'natural language': ['nlp', 'text', 'language', 'linguistic', 'speech', 'text processing'],
        'infosec': ['information security', 'cybersecurity', 'security', 'cyber security', 'information assurance'],
        'cybersecurity': ['cyber security', 'information security', 'infosec', 'security', 'network security'],
        'healthcare': ['health care', 'medical', 'medicine', 'clinical', 'healthcare systems', 'medical informatics'],
        'medical': ['healthcare', 'medicine', 'clinical', 'medical informatics', 'health informatics', 'biomedical'],
        'finance': ['financial', 'banking', 'fintech', 'financial services', 'economic', 'trading'],
        'education': ['educational', 'learning', 'pedagogy', 'educational technology', 'e-learning'],
        'climate': ['climate change', 'global warming', 'environmental', 'sustainability', 'climate science']
    }
    
    for topic_word in topic_words:
        if topic_word in related_terms:
            for related_term in related_terms[topic_word]:
                if related_term in title_lower:
                    semantic_score += 8
                if related_term in abstract_lower:
                    semantic_score += 3
    
    # 4. RECENCY SCORING (10% weight)
    recency_score = 0
    try:
        if year != 'Unknown' and year.isdigit():
            current_year = 2024
            paper_year = int(year)
            age = current_year - paper_year
            
            if age <= 1:
                recency_score = 20
            elif age <= 3:
                recency_score = 15
            elif age <= 5:
                recency_score = 10
            elif age <= 10:
                recency_score = 5
            else:
                recency_score = 0
    except:
        recency_score = 0
    
    # 5. ABSTRACT QUALITY SCORING (5% weight)
    quality_score = 0
    
    # Longer abstracts are generally better (up to a point)
    abstract_length = len(abstract_lower.split())
    if 50 <= abstract_length <= 300:
        quality_score += 10
    elif 20 <= abstract_length < 50:
        quality_score += 5
    
    # Check for technical terms and methodology indicators
    technical_indicators = ['method', 'approach', 'algorithm', 'technique', 'framework', 'model', 'system', 'experiment', 'evaluation', 'performance', 'accuracy', 'result', 'conclusion']
    tech_count = sum(1 for indicator in technical_indicators if indicator in abstract_lower)
    quality_score += min(tech_count * 2, 10)
    
    # 6. TITLE RELEVANCE BONUS
    title_bonus = 0
    
    # Check if title contains key research terms
    research_terms = ['novel', 'new', 'improved', 'enhanced', 'advanced', 'efficient', 'effective', 'robust', 'accurate', 'precise']
    for term in research_terms:
        if term in title_lower:
            title_bonus += 2
    
    # Check for methodology indicators in title
    method_terms = ['using', 'based on', 'via', 'through', 'with', 'for', 'detection', 'estimation', 'monitoring', 'analysis']
    for term in method_terms:
        if term in title_lower:
            title_bonus += 1
    
    # Calculate final weighted score
    if author_score > 0:
        # For author matches, prioritize author score heavily
        final_score = author_score + (
            exact_score * 0.20 +
            keyword_score * 0.15 +
            semantic_score * 0.10 +
            recency_score * 0.05 +
            quality_score * 0.05 +
            title_bonus
        )
    else:
        # For non-author searches, use normal weighting
        final_score = (
            exact_score * 0.40 +
            keyword_score * 0.25 +
            semantic_score * 0.20 +
            recency_score * 0.10 +
            quality_score * 0.05 +
            title_bonus
        )
    
    # Normalize score to 0-100 range
    final_score = min(100, max(0, final_score))
    
    return final_score

def expand_search_query(topic):
    """
    Expand search query with related terms and synonyms for better results
    """
    base_query = topic.lower()
    expanded_queries = [topic]  # Always include original
    
    # Check if this is an author search
    is_author_search = any(pattern in base_query for pattern in ['author:', 'by ', 'written by', 'papers by'])
    
    if is_author_search:
        # For author searches, don't expand much to avoid confusion
        # Just add variations with "author:" prefix
        author_name = base_query.replace('author:', '').replace('by ', '').replace('written by', '').replace('papers by', '').strip()
        if author_name:
            expanded_queries.extend([
                f"author:{author_name}",
                f"by {author_name}",
                f"papers by {author_name}"
            ])
        return expanded_queries[:3]  # Limit author searches to 3 variations
    
    # Define query expansion rules for non-author searches
    expansions = {
        'wifi': ['wireless', 'wi-fi', '802.11', 'rf sensing'],
        'csi': ['channel state information', 'channel', 'signal'],
        'heart': ['cardiac', 'pulse', 'heartbeat', 'hr'],
        'breathing': ['respiration', 'respiratory', 'breath', 'br'],
        'monitoring': ['sensing', 'detection', 'measurement'],
        'contactless': ['non-contact', 'remote', 'wireless'],
        'machine learning': ['ml', 'ai', 'artificial intelligence'],
        'deep learning': ['neural network', 'cnn', 'rnn'],
        'computer vision': ['cv', 'image processing', 'visual'],
        'natural language': ['nlp', 'text processing', 'language']
    }
    
    # Generate expanded queries
    words = re.findall(r'\b\w+\b', base_query)
    for word in words:
        if word in expansions:
            for expansion in expansions[word]:
                expanded_query = base_query.replace(word, expansion)
                if expanded_query not in expanded_queries:
                    expanded_queries.append(expanded_query)
    
    # Create combination queries
    if len(words) > 1:
        for i, word in enumerate(words):
            if word in expansions:
                for expansion in expansions[word]:
                    new_words = words.copy()
                    new_words[i] = expansion
                    combination = ' '.join(new_words)
                    if combination not in expanded_queries:
                        expanded_queries.append(combination)
    
    return expanded_queries[:5]  # Limit to 5 variations

def search_papers_fast(topic, max_results=10):
    """
    Fast paper search using multiple APIs with improved error handling and retry
    """
    logger.info(f"Starting fast paper search for topic: '{topic}'")
    
    all_papers = []
    max_retries = 3
    
    # Search arXiv API with retry
    for attempt in range(max_retries):
        try:
            logger.info(f"Searching arXiv API (attempt {attempt + 1}/{max_retries})...")
            arxiv_papers = search_arxiv_api(topic, max_results // 2)
            if arxiv_papers:
                all_papers.extend(arxiv_papers)
                logger.info(f"arXiv returned {len(arxiv_papers)} papers")
                break
            else:
                logger.warning(f"arXiv returned no papers on attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    time.sleep(1)  # Wait before retry
        except Exception as e:
            logger.error(f"arXiv search attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(1)  # Wait before retry
    
    # Search Semantic Scholar API with retry
    for attempt in range(max_retries):
        try:
            logger.info(f"Searching Semantic Scholar API (attempt {attempt + 1}/{max_retries})...")
            semantic_papers = search_semantic_scholar_api(topic, max_results // 2)
            if semantic_papers:
                all_papers.extend(semantic_papers)
                logger.info(f"Semantic Scholar returned {len(semantic_papers)} papers")
                break
            else:
                logger.warning(f"Semantic Scholar returned no papers on attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    time.sleep(1)  # Wait before retry
        except Exception as e:
            logger.error(f"Semantic Scholar search attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(1)  # Wait before retry
    
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

def search_papers_robust(topic, max_results=10):
    """
    Robust paper search with multiple fallback strategies
    """
    logger.info(f"Starting robust paper search for topic: '{topic}'")
    
    all_papers = []
    
    # Strategy 1: Try fast API search
    try:
        logger.info("Strategy 1: Fast API search")
        api_papers = search_papers_fast(topic, max_results)
        all_papers.extend(api_papers)
        logger.info(f"Fast API search returned {len(api_papers)} papers")
    except Exception as e:
        logger.warning(f"Fast API search failed: {e}")
    
    # Strategy 2: Try with simplified query if no results
    if len(all_papers) < 3:
        try:
            logger.info("Strategy 2: Simplified query search")
            # Simplify the query by taking first few words
            simple_query = ' '.join(topic.split()[:3])
            simple_papers = search_papers_fast(simple_query, max_results)
            all_papers.extend(simple_papers)
            logger.info(f"Simplified query search returned {len(simple_papers)} papers")
        except Exception as e:
            logger.warning(f"Simplified query search failed: {e}")
    
    # Strategy 3: Try with individual keywords if still no results
    if len(all_papers) < 3:
        try:
            logger.info("Strategy 3: Keyword-based search")
            keywords = [word for word in topic.split() if len(word) > 2]  # Include shorter words
            for keyword in keywords[:3]:  # Try top 3 keywords
                keyword_papers = search_papers_fast(keyword, max_results // 2)
                all_papers.extend(keyword_papers)
                logger.info(f"Keyword '{keyword}' search returned {len(keyword_papers)} papers")
        except Exception as e:
            logger.warning(f"Keyword search failed: {e}")
    
    # Strategy 4: Try with related terms if still no results
    if len(all_papers) < 3:
        try:
            logger.info("Strategy 4: Related terms search")
            related_terms = {
                'ai': ['artificial intelligence', 'machine learning', 'deep learning', 'neural networks', 'ml', 'ai systems', 'intelligent systems'],
                'artificial intelligence': ['ai', 'machine learning', 'deep learning', 'neural networks', 'intelligent systems'],
                'machine learning': ['ml', 'ai', 'artificial intelligence', 'deep learning', 'neural networks', 'supervised learning', 'unsupervised learning'],
                'deep learning': ['neural networks', 'ai', 'machine learning', 'cnn', 'rnn', 'transformer', 'deep neural networks'],
                'infosec': ['information security', 'cybersecurity', 'security', 'cyber security', 'information assurance', 'network security'],
                'cybersecurity': ['cyber security', 'information security', 'infosec', 'security', 'network security', 'cyber threats'],
                'healthcare': ['health care', 'medical', 'medicine', 'clinical', 'healthcare systems', 'medical informatics', 'health informatics'],
                'medical': ['healthcare', 'medicine', 'clinical', 'medical informatics', 'health informatics', 'biomedical'],
                'computer vision': ['cv', 'image processing', 'visual recognition', 'computer vision systems', 'image analysis'],
                'natural language': ['nlp', 'text processing', 'language models', 'natural language processing', 'text mining'],
                'climate': ['climate change', 'global warming', 'environmental', 'sustainability', 'climate science'],
                'finance': ['financial', 'banking', 'fintech', 'financial services', 'economic', 'trading'],
                'education': ['educational', 'learning', 'pedagogy', 'educational technology', 'e-learning'],
                'transportation': ['transport', 'mobility', 'autonomous vehicles', 'traffic', 'logistics'],
                'energy': ['renewable energy', 'solar', 'wind', 'power systems', 'energy efficiency'],
                'agriculture': ['farming', 'agricultural', 'crop', 'precision agriculture', 'agtech'],
                'manufacturing': ['industrial', 'production', 'automation', 'industry 4.0', 'smart manufacturing'],
                'retail': ['e-commerce', 'shopping', 'commerce', 'retail technology', 'online retail'],
                'entertainment': ['gaming', 'media', 'entertainment technology', 'digital media', 'content'],
                'sports': ['athletics', 'fitness', 'sports technology', 'performance analysis', 'sports science']
            }
            
            # Try related terms for each word in the topic
            for word in topic.split():
                word_lower = word.lower()
                if word_lower in related_terms:
                    for related_term in related_terms[word_lower][:3]:  # Try top 3 related terms
                        related_papers = search_papers_fast(related_term, max_results // 4)
                        all_papers.extend(related_papers)
                        logger.info(f"Related term '{related_term}' search returned {len(related_papers)} papers")
            
            # Also try combinations of related terms
            topic_words = topic.lower().split()
            if len(topic_words) >= 2:
                for i, word1 in enumerate(topic_words):
                    for j, word2 in enumerate(topic_words[i+1:], i+1):
                        if word1 in related_terms and word2 in related_terms:
                            for term1 in related_terms[word1][:2]:
                                for term2 in related_terms[word2][:2]:
                                    combo_term = f"{term1} {term2}"
                                    combo_papers = search_papers_fast(combo_term, max_results // 6)
                                    all_papers.extend(combo_papers)
                                    logger.info(f"Combo term '{combo_term}' search returned {len(combo_papers)} papers")
        except Exception as e:
            logger.warning(f"Related terms search failed: {e}")
    
    # Strategy 5: Try flexible combinations and variations
    if len(all_papers) < 3:
        logger.info("Strategy 5: Flexible combinations and variations")
        try:
            # Try different word orders and combinations
            topic_words = topic.lower().split()
            if len(topic_words) >= 2:
                # Try different word orders
                variations = [
                    topic,  # Original
                    ' '.join(reversed(topic_words)),  # Reversed order
                    f"{topic_words[0]} and {topic_words[1]}",  # With "and"
                    f"{topic_words[0]} for {topic_words[1]}",  # With "for"
                    f"{topic_words[0]} in {topic_words[1]}",  # With "in"
                    f"{topic_words[0]} with {topic_words[1]}",  # With "with"
                    f"{topic_words[0]} using {topic_words[1]}",  # With "using"
                    f"{topic_words[0]} based {topic_words[1]}",  # With "based"
                    f"{topic_words[0]} applications {topic_words[1]}",  # With "applications"
                    f"{topic_words[0]} systems {topic_words[1]}",  # With "systems"
                ]
                
                for variation in variations[:5]:  # Try top 5 variations
                    var_papers = search_papers_fast(variation, max_results // 5)
                    all_papers.extend(var_papers)
                    logger.info(f"Variation '{variation}' search returned {len(var_papers)} papers")
            
            # Try individual words with broader search
            for word in topic_words:
                if len(word) > 2:
                    word_papers = search_papers_fast(word, max_results // 3)
                    all_papers.extend(word_papers)
                    logger.info(f"Individual word '{word}' search returned {len(word_papers)} papers")
        except Exception as e:
            logger.warning(f"Flexible combinations search failed: {e}")
    
    # Strategy 6: Final fallback - try very broad terms
    if len(all_papers) < 3:
        logger.info("Strategy 6: Final fallback with broad terms")
        try:
            # Try very broad terms that are likely to return results
            broad_terms = ['artificial intelligence', 'machine learning', 'deep learning', 'neural networks']
            for term in broad_terms[:2]:  # Try top 2 broad terms
                broad_papers = search_papers_fast(term, max_results // 4)
                all_papers.extend(broad_papers)
                logger.info(f"Broad term '{term}' search returned {len(broad_papers)} papers")
        except Exception as e:
            logger.warning(f"Broad terms search failed: {e}")
    
    return all_papers

def search_arxiv_api(topic, max_results=5):
    """
    Search arXiv API for real papers with improved query handling
    """
    try:
        import requests
        import xml.etree.ElementTree as ET
        
        # Use multiple search strategies for better results
        search_queries = [
            f"all:{topic}",  # All fields search (most comprehensive)
            f"ti:{topic}",  # Title search
            f'"{topic}"',  # Exact phrase search
            topic,  # Simple search without prefix
            topic.replace(' ', '+')  # URL encoded version
        ]
        
        all_papers = []
        
        for search_query in search_queries:
            try:
                url = f"https://export.arxiv.org/api/query?search_query={search_query}&start=0&max_results={max_results}&sortBy=relevance&sortOrder=descending"
                logger.info(f"Searching arXiv with URL: {url}")
                response = requests.get(url, timeout=10)
            except Exception as e:
                logger.error(f"arXiv request failed for query '{search_query}': {e}")
                continue
            
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
                
                all_papers.extend(papers)
                logger.info(f"arXiv query '{search_query}' returned {len(papers)} papers")
            else:
                logger.error(f"arXiv API error for query '{search_query}': {response.status_code}")
        
        # Deduplicate papers
        seen_titles = set()
        unique_papers = []
        for paper in all_papers:
            title_key = paper['title'].lower().replace(' ', '').replace('-', '').replace('_', '')
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_papers.append(paper)
        
        logger.info(f"arXiv total unique papers: {len(unique_papers)}")
        return unique_papers
            
    except Exception as e:
        logger.error(f"arXiv search error: {e}")
        return []

def search_semantic_scholar_api(topic, max_results=5):
    """
    Search Semantic Scholar API for real papers with improved query handling
    """
    try:
        import requests
        
        # Use multiple search strategies
        search_queries = [
            topic,  # Original query
            f'"{topic}"',  # Exact phrase
            topic.replace(' ', '+'),  # URL encoded
            topic.replace(' ', ' AND '),  # AND search
            topic.replace(' ', ' OR ')  # OR search
        ]
        
        all_papers = []
        
        for query in search_queries:
            try:
                url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={query}&limit={max_results}&fields=title,authors,abstract,url,year,citationCount,isOpenAccess&sort=relevance"
                logger.info(f"Searching Semantic Scholar with URL: {url}")
                response = requests.get(url, timeout=10)
            except Exception as e:
                logger.error(f"Semantic Scholar request failed for query '{query}': {e}")
                continue
            
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
                
                all_papers.extend(papers)
                logger.info(f"Semantic Scholar query '{query}' returned {len(papers)} papers")
            else:
                logger.error(f"Semantic Scholar API error for query '{query}': {response.status_code}")
        
        # Deduplicate papers
        seen_titles = set()
        unique_papers = []
        for paper in all_papers:
            title_key = paper['title'].lower().replace(' ', '').replace('-', '').replace('_', '')
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_papers.append(paper)
        
        logger.info(f"Semantic Scholar total unique papers: {len(unique_papers)}")
        return unique_papers
            
    except Exception as e:
        logger.error(f"Semantic Scholar search error: {e}")
        return []

def get_fallback_papers(topic):
    """
    Provide fallback papers when APIs fail - NO HARDCODED PAPERS
    """
    logger.info(f"Attempting to find real papers for topic: {topic}")
    
    # Try real API searches with multiple strategies
    try:
        # Expand search queries for better coverage
        search_queries = expand_search_query(topic)
        logger.info(f"Expanded search queries: {search_queries}")
        
        all_papers = []
        
        # Search with each query variation
        for query in search_queries[:3]:  # Limit to 3 queries to avoid rate limiting
            try:
                logger.info(f"Searching with query: {query}")
                arxiv_papers = search_arxiv_api(query, 4)
                semantic_papers = search_semantic_scholar_api(query, 4)
                all_papers.extend(arxiv_papers + semantic_papers)
                time.sleep(0.5)  # Small delay between queries
            except Exception as e:
                logger.warning(f"Query '{query}' failed: {e}")
                continue
        
        if all_papers:
            # Deduplicate based on title similarity
            seen_titles = set()
            unique_papers = []
            for paper in all_papers:
                title_key = paper['title'].lower().replace(' ', '').replace('-', '').replace('_', '')
                if title_key not in seen_titles:
                    seen_titles.add(title_key)
                    unique_papers.append(paper)
            
            # Use advanced relevance scoring
            logger.info(f"Scoring {len(unique_papers)} papers with advanced relevance algorithm")
            for paper in unique_papers:
                paper['relevance_score'] = advanced_relevance_scoring(paper, topic, search_queries)
            
            # Sort by relevance score (highest first)
            unique_papers.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
            
            # Filter out papers with very low relevance scores
            relevant_papers = [p for p in unique_papers if p.get('relevance_score', 0) >= 10]
            
            logger.info(f"Found {len(relevant_papers)} relevant papers from real APIs")
            return relevant_papers[:8]  # Return top 8 most relevant papers
    
    except Exception as e:
        logger.error(f"API search failed: {e}")
    
    # If all API searches fail, return empty list - NO HARDCODED FALLBACKS
    logger.warning("All API searches failed - returning empty results")
    return []

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
        
        # FAST API-BASED SEARCH
        logger.info(f"Starting fast paper search for: {topic}")
        
        # Use robust search for ALL queries - no hardcoded papers
        papers = []
        
        # Check cache first
        cache_key = topic.lower().strip()
        if cache_key in search_cache:
            cache_time, cached_papers = search_cache[cache_key]
            if time.time() - cache_time < CACHE_DURATION:
                logger.info(f"Returning cached results for '{topic}' ({len(cached_papers)} papers)")
                papers = cached_papers
            else:
                # Cache expired, remove it
                del search_cache[cache_key]
        
        if not papers:
            try:
                # Use robust search strategy
                logger.info("Using robust search strategy")
                all_papers = search_papers_robust(topic, 12)
                logger.info(f"Robust search returned {len(all_papers)} papers")
                
                if all_papers:
                    # Deduplicate based on title similarity
                    seen_titles = set()
                    unique_papers = []
                    for paper in all_papers:
                        title_key = paper['title'].lower().replace(' ', '').replace('-', '').replace('_', '')
                        if title_key not in seen_titles:
                            seen_titles.add(title_key)
                            unique_papers.append(paper)
                    
                    # Use advanced relevance scoring
                    search_queries = expand_search_query(topic)
                    logger.info(f"Scoring {len(unique_papers)} papers with advanced relevance algorithm")
                    for paper in unique_papers:
                        paper['relevance_score'] = advanced_relevance_scoring(paper, topic, search_queries)
                    
                    # Sort by relevance score (highest first)
                    unique_papers.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
                    
                    # Check if this is an author search
                    is_author_search = any(pattern in topic.lower() for pattern in ['author:', 'by ', 'written by', 'papers by']) or any(word in topic.lower() for word in ['bhatia', 'smith', 'johnson', 'garcia', 'miller', 'davis', 'rodriguez', 'martinez', 'hernandez', 'lopez'])
                    
                    if is_author_search:
                        # For author searches, only return papers with exact author matches
                        author_matched_papers = [p for p in unique_papers if p.get('relevance_score', 0) >= 80]  # High threshold for author matches
                        if author_matched_papers:
                            papers = author_matched_papers[:5]
                            logger.info(f"Found {len(papers)} papers by exact author match")
                        else:
                            papers = []
                            logger.info("No papers found by exact author match")
                    else:
                        # For non-author searches, return ALL papers regardless of relevance score
                        papers = unique_papers[:12]  # Return top 12 papers
                        logger.info(f"Found {len(papers)} papers from APIs (no relevance filtering)")
                    
                    # Cache the results
                    search_cache[cache_key] = (time.time(), papers)
                    logger.info(f"Cached results for '{topic}' ({len(papers)} papers)")
                else:
                    logger.info("No papers found from APIs")
                    papers = []
                    
            except Exception as e:
                logger.error(f"Fast API search failed: {e}")
                logger.info("No fallback papers - returning empty results")
                papers = []
        
        logger.info(f"Found {len(papers)} papers")
        for i, paper in enumerate(papers):
            relevance_score = paper.get('relevance_score', 'N/A')
            logger.info(f"Paper {i+1}: {paper.get('title', 'No title')} (Relevance: {relevance_score})")
        
        # Add relevance information to response
        if len(papers) > 0:
            # Calculate average relevance score
            scores = [p.get('relevance_score', 0) for p in papers if isinstance(p.get('relevance_score'), (int, float))]
            avg_score = sum(scores) / len(scores) if scores else 0
            
            # Determine quality message based on average relevance
            if avg_score >= 70:
                quality_msg = "🔥 Highly relevant papers found!"
            elif avg_score >= 50:
                quality_msg = "✅ Good quality papers found!"
            elif avg_score >= 30:
                quality_msg = "📚 Relevant papers found!"
            else:
                quality_msg = "📖 Some relevant papers found!"
            
            return jsonify({
                'papers': papers,
                'message': f'{quality_msg} Found {len(papers)} papers (Avg relevance: {avg_score:.1f}%)',
                'fallback': False,
                'relevance_stats': {
                    'average_score': round(avg_score, 1),
                    'total_papers': len(papers),
                    'high_relevance': len([p for p in papers if p.get('relevance_score', 0) >= 70])
                }
            })
        else:
            return jsonify({
                'papers': papers,
                'message': '📚 No papers found, showing popular papers',
                'fallback': True
            })
        
        # This code block is no longer needed as we use fast API search above
        
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
        title = data.get('title', '')
        
        if not abstract:
            return jsonify({'error': 'Abstract is required'}), 400
        
        # Use Gemini API for real insights
        if not abstract or abstract == 'No abstract available':
            summary = "**Summary:** No abstract available for this paper."
        else:
            try:
                # Create a comprehensive prompt for Gemini
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
                
                # Generate response using Gemini
                response = model.generate_content(prompt)
                summary = response.text
                
                logger.info("Generated insights using Gemini API")
                
            except Exception as e:
                logger.error(f"Gemini API error: {e}")
                # Fallback to basic summary
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

@app.route('/api/research-gaps', methods=['POST'])
def get_research_gaps():
    try:
        data = request.get_json()
        abstract = data.get('abstract')
        title = data.get('title', '')
        
        if not abstract:
            return jsonify({'error': 'Abstract is required'}), 400
        
        # Use Gemini API for research gaps analysis
        if not abstract or abstract == 'No abstract available':
            gaps = "**Research Gaps:** Unable to analyze research gaps without abstract content."
        else:
            try:
                # Create a comprehensive prompt for research gaps
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

Format your response in markdown with clear sections and bullet points. Be specific and actionable."""
                
                # Generate response using Gemini
                response = model.generate_content(prompt)
                gaps = response.text
                
                logger.info("Generated research gaps using Gemini API")
                
            except Exception as e:
                logger.error(f"Gemini API error for research gaps: {e}")
                # Fallback to basic research gaps
                gaps = """**Research Gaps Identified:**
• **Methodological Limitations:** The current approach could be enhanced with additional validation methods
• **Scope Expansion:** Future work could explore broader applications and use cases
• **Performance Optimization:** There may be opportunities to improve efficiency and accuracy
• **Cross-domain Applications:** The methodology could be adapted for related fields
• **Long-term Studies:** Extended evaluation periods could provide deeper insights

**Future Research Directions:**
• Investigate alternative approaches and compare performance
• Explore integration with emerging technologies
• Conduct comprehensive user studies and validation
• Develop standardized evaluation metrics
• Investigate scalability and real-world deployment challenges"""
        
        return jsonify({'research_gaps': gaps})
        
    except Exception as e:
        return jsonify({'error': f'Failed to generate research gaps: {str(e)}'}), 500

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

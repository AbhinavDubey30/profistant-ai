// Real-time paper search using multiple APIs
const searchArxiv = async (query, maxResults = 5) => {
  try {
    const response = await fetch(`https://export.arxiv.org/api/query?search_query=all:${encodeURIComponent(query)}&start=0&max_results=${maxResults}&sortBy=relevance&sortOrder=descending`);
    const xmlText = await response.text();
    
    // Parse XML response
    const parser = new DOMParser();
    const xmlDoc = parser.parseFromString(xmlText, 'text/xml');
    const entries = xmlDoc.getElementsByTagName('entry');
    
    const papers = [];
    for (let i = 0; i < entries.length; i++) {
      const entry = entries[i];
      const title = entry.getElementsByTagName('title')[0]?.textContent?.trim() || 'No title';
      const summary = entry.getElementsByTagName('summary')[0]?.textContent?.trim() || 'No abstract available';
      const authors = Array.from(entry.getElementsByTagName('author')).map(author => 
        author.getElementsByTagName('name')[0]?.textContent || 'Unknown'
      ).join(', ');
      const link = entry.getElementsByTagName('link')[0]?.getAttribute('href') || '#';
      const published = entry.getElementsByTagName('published')[0]?.textContent?.substring(0, 4) || 'Unknown';
      
      papers.push({
        title,
        authors,
        abstract: summary,
        url: link,
        year: published
      });
    }
    
    return papers;
  } catch (error) {
    console.error('arXiv search error:', error);
    return [];
  }
};

const searchSemanticScholar = async (query, maxResults = 5) => {
  try {
    const response = await fetch(`https://api.semanticscholar.org/graph/v1/paper/search?query=${encodeURIComponent(query)}&limit=${maxResults}&fields=title,authors,abstract,url,year`);
    const data = await response.json();
    
    if (data.data) {
      return data.data.map(paper => ({
        title: paper.title || 'No title',
        authors: paper.authors?.map(author => author.name).join(', ') || 'Unknown',
        abstract: paper.abstract || 'No abstract available',
        url: paper.url || '#',
        year: paper.year?.toString() || 'Unknown'
      }));
    }
    
    return [];
  } catch (error) {
    console.error('Semantic Scholar search error:', error);
    return [];
  }
};

const searchCrossRef = async (query, maxResults = 5) => {
  try {
    const response = await fetch(`https://api.crossref.org/works?query=${encodeURIComponent(query)}&rows=${maxResults}&sort=relevance`);
    const data = await response.json();
    
    if (data.message && data.message.items) {
      return data.message.items.map(paper => ({
        title: paper.title?.[0] || 'No title',
        authors: paper.author?.map(author => `${author.given} ${author.family}`).join(', ') || 'Unknown',
        abstract: paper.abstract || 'No abstract available',
        url: paper.URL || '#',
        year: paper['published-print']?.['date-parts']?.[0]?.[0]?.toString() || 
              paper['published-online']?.['date-parts']?.[0]?.[0]?.toString() || 'Unknown'
      }));
    }
    
    return [];
  } catch (error) {
    console.error('CrossRef search error:', error);
    return [];
  }
};

// Function to deduplicate papers based on title similarity
const deduplicatePapers = (papers) => {
  const seen = new Set();
  return papers.filter(paper => {
    const titleKey = paper.title.toLowerCase().replace(/[^\w\s]/g, '').trim();
    if (seen.has(titleKey)) {
      return false;
    }
    seen.add(titleKey);
    return true;
  });
};

// Function to calculate relevance score for sorting
const calculateRelevanceScore = (paper, query) => {
  const queryLower = query.toLowerCase();
  const titleLower = paper.title.toLowerCase();
  const abstractLower = paper.abstract.toLowerCase();
  
  let score = 0;
  
  // Title exact match gets highest score
  if (titleLower.includes(queryLower)) {
    score += 10;
  }
  
  // Abstract contains query gets medium score
  if (abstractLower.includes(queryLower)) {
    score += 5;
  }
  
  // Word-by-word matching
  const queryWords = queryLower.split(/\s+/).filter(word => word.length > 2);
  queryWords.forEach(word => {
    if (titleLower.includes(word)) {
      score += 3;
    }
    if (abstractLower.includes(word)) {
      score += 1;
    }
  });
  
  return score;
};

// Main search function that tries multiple APIs
export const getInstantPapers = async (topic) => {
  if (!topic || topic.trim() === '') {
    return [];
  }
  
  console.log(`🔍 Searching for papers on: "${topic}"`);
  
  try {
    // Try multiple APIs in parallel
    const [arxivResults, semanticResults, crossrefResults] = await Promise.allSettled([
      searchArxiv(topic, 3),
      searchSemanticScholar(topic, 3),
      searchCrossRef(topic, 3)
    ]);
    
    // Combine results from all APIs
    let allPapers = [];
    
    if (arxivResults.status === 'fulfilled') {
      allPapers = [...allPapers, ...arxivResults.value];
    }
    
    if (semanticResults.status === 'fulfilled') {
      allPapers = [...allPapers, ...semanticResults.value];
    }
    
    if (crossrefResults.status === 'fulfilled') {
      allPapers = [...allPapers, ...crossrefResults.value];
    }
    
    // Deduplicate and sort by relevance
    const uniquePapers = deduplicatePapers(allPapers);
    const papersWithScores = uniquePapers.map(paper => ({
      ...paper,
      relevanceScore: calculateRelevanceScore(paper, topic)
    }));
    
    // Sort by relevance and return top 5
    const sortedPapers = papersWithScores
      .sort((a, b) => b.relevanceScore - a.relevanceScore)
      .slice(0, 5);
    
    console.log(`✅ Found ${sortedPapers.length} relevant papers`);
    
    // If no papers found, return fallback papers
    if (sortedPapers.length === 0) {
      console.log('⚠️ No papers found, using fallback');
      return getFallbackPapers();
    }
    
    return sortedPapers;
    
  } catch (error) {
    console.error('Search error:', error);
    console.log('⚠️ Using fallback papers due to error');
    return getFallbackPapers();
  }
};

// Fallback papers for when APIs fail
const getFallbackPapers = () => {
  return [
    {
      title: "Attention Is All You Need",
      authors: "Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, L., Polosukhin, I.",
      abstract: "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.",
      url: "https://arxiv.org/abs/1706.03762",
      year: "2017"
    },
    {
      title: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
      authors: "Devlin, J., Chang, M.W., Lee, K., Toutanova, K.",
      abstract: "We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers.",
      url: "https://arxiv.org/abs/1810.04805",
      year: "2018"
    },
    {
      title: "Deep Residual Learning for Image Recognition",
      authors: "He, K., Zhang, X., Ren, S., Sun, J.",
      abstract: "Deeper neural networks are more difficult to train. We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously. We explicitly reformulate the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions.",
      url: "https://arxiv.org/abs/1512.03385",
      year: "2015"
    },
    {
      title: "Generative Adversarial Networks",
      authors: "Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., Bengio, Y.",
      abstract: "We propose a new framework for estimating generative models via an adversarial process, in which we simultaneously train two models: a generative model G that captures the data distribution, and a discriminative model D that estimates the probability that a sample came from the training data rather than G.",
      url: "https://arxiv.org/abs/1406.2661",
      year: "2014"
    },
    {
      title: "Deep Learning",
      authors: "LeCun, Y., Bengio, Y., Hinton, G.",
      abstract: "Deep learning allows computational models that are composed of multiple processing layers to learn representations of data with multiple levels of abstraction. These methods have dramatically improved the state-of-the-art in speech recognition, visual object recognition, object detection and many other domains.",
      url: "https://www.nature.com/articles/nature14539",
      year: "2015"
    }
  ];
};

export const getInstantSummary = (abstract) => {
  return `**Key Points:**
• This research explores ${abstract.substring(0, 50)}...
• The study presents novel methodologies and approaches
• Results demonstrate significant improvements in the field

**Research Direction:** Future work could investigate the application of these methods to real-world scenarios and compare performance with existing approaches.`;
};

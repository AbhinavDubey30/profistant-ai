import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || (process.env.NODE_ENV === 'production' ? 'https://profistant-ai-production.up.railway.app/api' : 'http://localhost:5000/api');

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60000, // Increased to 60 seconds
});

// Test connection to backend
export const testConnection = async () => {
  try {
    const response = await api.get('/test');
    console.log('Backend connection test:', response.data);
    return true;
  } catch (error) {
    console.error('Backend connection test failed:', error);
    return false;
  }
};

export const searchPapers = async (topic, settings) => {
  try {
    console.log('🔍 API: Searching papers for:', topic);
    console.log('🔍 API: Base URL:', api.defaults.baseURL);
    console.log('🔍 API: Full endpoint:', `${api.defaults.baseURL}/search-papers`);
    
    const response = await api.post('/search-papers', {
      topic,
      settings
    }, {
      timeout: 25000 // 25 seconds for reliable results
    });
    
    console.log('📊 API: Response status:', response.status);
    console.log('📊 API: Response data:', response.data);
    console.log('📊 API: Papers in response:', response.data.papers);
    console.log('📊 API: Papers count:', response.data.papers?.length);
    
    // Handle fallback results
    if (response.data.fallback) {
      console.warn('⚠️ API: Using fallback results:', response.data.message);
    }
    
    const result = {
      papers: response.data.papers,
      isFallback: response.data.fallback || false,
      message: response.data.message || null
    };
    
    console.log('📊 API: Returning result:', result);
    return result;
  } catch (error) {
    console.error('Search error:', error);
    
    if (error.code === 'ECONNABORTED' || error.message.includes('timeout')) {
      throw new Error('Search is taking too long. Please try again with a more specific search term.');
    }
    
    if (error.response?.data?.error) {
      throw new Error(error.response.data.error);
    }
    
    if (error.code === 'ECONNREFUSED') {
      throw new Error('Cannot connect to server. Please make sure the backend is running.');
    }
    
    if (error.message.includes('Network Error')) {
      throw new Error('Network error. Please check your connection and try again.');
    }
    
    throw new Error(`Failed to search papers: ${error.message}`);
  }
};

export const summarizePaper = async (abstract, title = '') => {
  try {
    const response = await api.post('/summarize', {
      abstract,
      title
    });
    return response.data.summary;
  } catch (error) {
    if (error.response?.data?.error) {
      throw new Error(error.response.data.error);
    }
    throw new Error('Failed to generate summary. Please try again.');
  }
};

export const getResearchGaps = async (abstract, title = '') => {
  try {
    const response = await api.post('/research-gaps', {
      abstract,
      title
    });
    return response.data.research_gaps;
  } catch (error) {
    if (error.response?.data?.error) {
      throw new Error(error.response.data.error);
    }
    throw new Error('Failed to generate research gaps. Please try again.');
  }
};

export const getReadingList = async () => {
  try {
    const response = await api.get('/reading-list');
    return response.data.papers;
  } catch (error) {
    throw new Error('Failed to load reading list');
  }
};

export const saveToReadingList = async (paper) => {
  try {
    const response = await api.post('/reading-list', paper);
    return response.data;
  } catch (error) {
    throw new Error('Failed to save paper');
  }
};

export const generateCalendar = async (readingList, totalWeeks, selectedDays, timeRange) => {
  try {
    const response = await api.post('/calendar-export', {
      readingList,
      totalWeeks,
      selectedDays,
      timeRange
    });
    return response.data;
  } catch (error) {
    if (error.response?.data?.error) {
      throw new Error(error.response.data.error);
    }
    throw new Error('Failed to generate calendar');
  }
};

export const generateResearchPlan = async (readingList, totalWeeks, selectedDays, timeRange) => {
  try {
    const response = await api.post('/research-plan', {
      readingList,
      totalWeeks,
      selectedDays,
      timeRange
    });
    return response.data;
  } catch (error) {
    if (error.response?.data?.error) {
      throw new Error(error.response.data.error);
    }
    throw new Error('Failed to generate research plan');
  }
};

export const generateResearchIdeas = async (readingList) => {
  try {
    const response = await api.post('/research-ideas', {
      readingList
    });
    return response.data;
  } catch (error) {
    if (error.response?.data?.error) {
      throw new Error(error.response.data.error);
    }
    throw new Error('Failed to generate research ideas');
  }
};

export default api;

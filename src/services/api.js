import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
});

// Test connection to backend
export const testConnection = async () => {
  try {
    const response = await api.get('/api/test');
    console.log('Backend connection test:', response.data);
    return true;
  } catch (error) {
    console.error('Backend connection test failed:', error);
    return false;
  }
};

export const searchPapers = async (topic, settings) => {
  try {
    console.log('Searching papers for:', topic);
    const response = await api.post('/api/search-papers', {
      topic,
      settings
    });
    console.log('Search response:', response.data);
    return response.data.papers;
  } catch (error) {
    console.error('Search error:', error);
    if (error.response?.data?.error) {
      throw new Error(error.response.data.error);
    }
    if (error.code === 'ECONNREFUSED') {
      throw new Error('Cannot connect to server. Please make sure the backend is running on port 5000.');
    }
    if (error.message.includes('Network Error')) {
      throw new Error('Network error. Please check your connection and try again.');
    }
    throw new Error(`Failed to search papers: ${error.message}`);
  }
};

export const summarizePaper = async (abstract) => {
  try {
    const response = await api.post('/api/summarize', {
      abstract
    });
    return response.data.summary;
  } catch (error) {
    if (error.response?.data?.error) {
      throw new Error(error.response.data.error);
    }
    throw new Error('Failed to generate summary. Please try again.');
  }
};

export const getReadingList = async () => {
  try {
    const response = await api.get('/api/reading-list');
    return response.data.papers;
  } catch (error) {
    throw new Error('Failed to load reading list');
  }
};

export const saveToReadingList = async (paper) => {
  try {
    const response = await api.post('/api/reading-list', paper);
    return response.data;
  } catch (error) {
    throw new Error('Failed to save paper');
  }
};

export default api;

import React, { useState, useEffect } from 'react';
import { Container, Row, Col, Card, Form, Button, Alert, ProgressBar, Spinner } from 'react-bootstrap';
import { FaSearch, FaBookOpen, FaPlus, FaCog, FaTrash } from 'react-icons/fa';
import { searchPapers, summarizePaper, testConnection } from '../services/api';
import { getInstantSummary } from '../utils/instantSearch';

const Home = ({ readingList, addToReadingList, settings, setSettings }) => {
  const [topic, setTopic] = useState('');
  const [papers, setPapers] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [progress, setProgress] = useState(0);
  const [statusText, setStatusText] = useState('');
  const [showSettings, setShowSettings] = useState(false);
  const [manualMode, setManualMode] = useState(false);
  const [manualPaper, setManualPaper] = useState({
    title: '',
    authors: '',
    abstract: '',
    url: ''
  });
  const [connectionStatus, setConnectionStatus] = useState('checking');
  const [expandedSummary, setExpandedSummary] = useState(null);

  // Test backend connection on component mount
  useEffect(() => {
    const checkConnection = async () => {
      console.log('🔍 Testing backend connection...');
      const isConnected = await testConnection();
      console.log('🔍 Backend connection status:', isConnected ? 'CONNECTED' : 'DISCONNECTED');
      setConnectionStatus(isConnected ? 'connected' : 'disconnected');
    };
    checkConnection();
  }, []);

  const handleSearch = async () => {
    console.log('🚀 Starting search process...');
    console.log('📝 Topic:', topic);
    console.log('⚙️ Settings:', settings);
    console.log('🔗 Connection Status:', connectionStatus);
    
    if (!topic.trim()) {
      console.log('❌ No topic provided');
      setError('Please enter a research topic');
      return;
    }

    console.log('✅ Starting REAL-TIME search...');
    setLoading(true);
    setError('');
    setSuccess('');
    setProgress(20);
    setStatusText('🔍 Searching arXiv...');

    try {
      // REAL-TIME SEARCH using backend API
      console.log('🔍 Searching real papers via backend API...');
      
      setProgress(40);
      setStatusText('🔍 Searching multiple APIs...');
      
      setProgress(60);
      setStatusText('🔍 Processing results...');
      
      setProgress(80);
      setStatusText('📊 Finalizing search...');
      
      // Use the backend API instead of instant search
      const response = await searchPapers(topic, settings);
      setPapers(response.papers);
      
      setProgress(100);
      setStatusText('✅ Search completed!');
      
      if (response.papers.length > 0) {
        setSuccess(`🎉 ${response.message}`);
      } else {
        setSuccess('📚 No papers found');
      }
    } catch (err) {
      console.error('❌ Search failed:', err);
      console.error('❌ Error details:', {
        message: err.message,
        stack: err.stack,
        name: err.name
      });
      setError(err.message || 'Failed to search papers');
    } finally {
      console.log('🏁 Search process completed');
      setLoading(false);
      setProgress(0);
      setStatusText('');
    }
  };

  const handleSummarize = async (paper) => {
    setLoading(true);
    setStatusText('📖 Generating full summary...');
    
    try {
      // Generate full summary with complete abstract
      await new Promise(resolve => setTimeout(resolve, 300));
      const summary = getInstantSummary(paper.abstract);
      
      // Set the expanded summary to show inline
      setExpandedSummary({
        paper: paper,
        summary: summary
      });
      
      setSuccess(`📖 Full summary generated for "${paper.title}"`);
    } catch (err) {
      setError('Failed to generate summary');
    } finally {
      setLoading(false);
      setStatusText('');
    }
  };

  const handleAddToReadingList = (paper) => {
    addToReadingList(paper);
    setSuccess(`Added "${paper.title}" to reading list!`);
  };

  const handleManualAdd = () => {
    if (!manualPaper.title || !manualPaper.authors || !manualPaper.abstract) {
      setError('Please fill in title, authors, and abstract');
      return;
    }
    
    addToReadingList(manualPaper);
    setManualPaper({ title: '', authors: '', abstract: '', url: '' });
    setManualMode(false);
    setSuccess('Paper added to reading list!');
  };

  return (
    <div className="research-page">
      <Container fluid className="px-4">
        <div className="text-center mb-4">
          <h1 className="page-title">
            <FaBookOpen className="me-3" />
            Profistant - Your Research Kick-Starter
          </h1>
          <p className="text-white-50 fs-6">
            AI-powered research assistant designed to help you kick-start your academic journey
          </p>
        </div>

        <Row className="g-3">
          <Col lg={12}>
          <Card className="search-card">
            <Card.Body className="p-4">
              <h4 className="card-title mb-3">
                <FaSearch className="me-2 text-primary" />
                Search Research Papers
              </h4>
              
              <Form.Group className="mb-3">
                <Form.Control
                  type="text"
                  placeholder="Enter your research topic or area of interest..."
                  value={topic}
                  onChange={(e) => setTopic(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
                  className="form-control-lg"
                />
              </Form.Group>

              <div className="d-flex gap-2 mb-3">
                <Button 
                  variant="primary" 
                  onClick={handleSearch}
                  disabled={loading}
                  className="flex-fill"
                  size="lg"
                >
                  {loading ? (
                    <>
                      <Spinner size="sm" className="me-2" />
                      Searching...
                    </>
                  ) : (
                    <>
                      <FaSearch className="me-2" />
                      Search Papers
                    </>
                  )}
                </Button>
                <Button 
                  variant="outline-secondary"
                  onClick={() => setShowSettings(!showSettings)}
                  size="lg"
                >
                  <FaCog />
                </Button>
              </div>

              {loading && (
                <div className="mb-3">
                  <ProgressBar now={progress} className="mb-2" />
                  <small className="text-muted">{statusText}</small>
                </div>
              )}

              {/* Connection Status */}
              <div className="mb-3">
                <small className={`badge ${connectionStatus === 'connected' ? 'bg-success' : connectionStatus === 'disconnected' ? 'bg-danger' : 'bg-warning'}`}>
                  {connectionStatus === 'connected' ? '🟢 Backend Connected' : 
                   connectionStatus === 'disconnected' ? '🔴 Backend Disconnected' : 
                   '🟡 Checking Connection...'}
                </small>
              </div>

              {error && <Alert variant="danger">{error}</Alert>}
              {success && (
                <Alert variant="success" style={{ whiteSpace: 'pre-line' }}>
                  {success}
                </Alert>
              )}

              {/* Expanded Summary Display */}
              {expandedSummary && (
                <Card className="mt-3">
                  <Card.Body>
                    <div className="d-flex justify-content-between align-items-center mb-3">
                      <h5 className="mb-0">📖 Full Summary: {expandedSummary.paper.title}</h5>
                      <Button 
                        variant="outline-secondary" 
                        size="sm"
                        onClick={() => setExpandedSummary(null)}
                      >
                        ✕ Close
                      </Button>
                    </div>
                    <div 
                      style={{ 
                        maxHeight: '500px', 
                        overflowY: 'auto', 
                        padding: '15px', 
                        backgroundColor: '#f8f9fa', 
                        borderRadius: '5px',
                        whiteSpace: 'pre-line',
                        fontSize: '14px',
                        lineHeight: '1.6'
                      }}
                    >
                      {expandedSummary.summary}
                    </div>
                  </Card.Body>
                </Card>
              )}

              {showSettings && (
                <Card className="mt-3">
                  <Card.Body className="p-3">
                    <h6 className="mb-3">Advanced Settings</h6>
                    <Row>
                      <Col md={6}>
                        <Form.Group className="mb-3">
                          <Form.Label className="small">Timeout (seconds)</Form.Label>
                          <Form.Range
                            min="5"
                            max="30"
                            value={settings.timeout}
                            onChange={(e) => setSettings({...settings, timeout: parseInt(e.target.value)})}
                          />
                          <small className="text-muted">{settings.timeout}s</small>
                        </Form.Group>
                      </Col>
                      <Col md={6}>
                        <Form.Group className="mb-3">
                          <Form.Check
                            type="checkbox"
                            label="Use Proxy"
                            checked={settings.useProxy}
                            onChange={(e) => setSettings({...settings, useProxy: e.target.checked})}
                          />
                        </Form.Group>
                      </Col>
                    </Row>
                  </Card.Body>
                </Card>
              )}

              {papers.length > 0 && (
                <div className="mt-3">
                  <div className="d-flex justify-content-between align-items-center mb-3">
                    <h5 className="mb-0">Found Papers:</h5>
                    <div className="d-flex align-items-center gap-2">
                      <small className="text-muted">
                        Sorted by relevance (most relevant first)
                      </small>
                    </div>
                  </div>
                  <div className="papers-container">
                    {papers.map((paper, index) => {
                      const relevanceScore = paper.relevance_score || 0;
                      const getRelevanceColor = (score) => {
                        if (score >= 70) return 'success';
                        if (score >= 50) return 'warning';
                        if (score >= 30) return 'info';
                        return 'secondary';
                      };
                      const getRelevanceLabel = (score) => {
                        if (score >= 70) return '🔥 Highly Relevant';
                        if (score >= 50) return '✅ Very Relevant';
                        if (score >= 30) return '📚 Relevant';
                        return '📖 Somewhat Relevant';
                      };
                      
                      return (
                        <Card key={index} className="paper-card mb-3">
                          <Card.Body className="p-3">
                            <div className="d-flex justify-content-between align-items-start mb-2">
                              <h6 className="card-title mb-0 flex-grow-1">{paper.title}</h6>
                              {paper.relevance_score && (
                                <span className={`badge bg-${getRelevanceColor(relevanceScore)} ms-2`}>
                                  {getRelevanceLabel(relevanceScore)} ({relevanceScore.toFixed(0)}%)
                                </span>
                              )}
                            </div>
                            <p className="text-muted mb-2 small">
                              <strong>Authors:</strong> {paper.authors}
                              {paper.year && paper.year !== 'Unknown' && (
                                <span className="ms-2">
                                  <strong>Year:</strong> {paper.year}
                                </span>
                              )}
                            </p>
                            <p className="card-text small mb-3">{paper.abstract}</p>
                            <div className="d-flex gap-2 flex-wrap">
                              <Button 
                                variant="outline-primary" 
                                size="sm"
                                onClick={() => handleSummarize(paper)}
                                disabled={loading}
                              >
                                Summarize
                              </Button>
                              <Button 
                                variant="outline-success" 
                                size="sm"
                                onClick={() => handleAddToReadingList(paper)}
                              >
                                <FaPlus className="me-1" />
                                Add to List
                              </Button>
                              {paper.url && paper.url !== '#' && (
                                <Button 
                                  variant="outline-info" 
                                  size="sm"
                                  href={paper.url}
                                  target="_blank"
                                >
                                  View Paper
                                </Button>
                              )}
                            </div>
                          </Card.Body>
                        </Card>
                      );
                    })}
                  </div>
                </div>
              )}

              {error && !loading && (
                <div className="mt-3">
                  <h6>Alternative Options:</h6>
                  <div className="d-flex gap-2">
                    <Button 
                      variant="outline-warning" 
                      onClick={handleSearch}
                      size="sm"
                    >
                      Try Again
                    </Button>
                    <Button 
                      variant="outline-info"
                      onClick={() => setManualMode(true)}
                      size="sm"
                    >
                      Manual Input
                    </Button>
                  </div>
                </div>
              )}

              {manualMode && (
                <Card className="mt-3">
                  <Card.Body className="p-3">
                    <h6 className="mb-3">Manual Paper Input</h6>
                    <Row>
                      <Col md={6}>
                        <Form.Group className="mb-3">
                          <Form.Label className="small">Paper Title</Form.Label>
                          <Form.Control
                            type="text"
                            value={manualPaper.title}
                            onChange={(e) => setManualPaper({...manualPaper, title: e.target.value})}
                            size="sm"
                          />
                        </Form.Group>
                      </Col>
                      <Col md={6}>
                        <Form.Group className="mb-3">
                          <Form.Label className="small">Authors</Form.Label>
                          <Form.Control
                            type="text"
                            value={manualPaper.authors}
                            onChange={(e) => setManualPaper({...manualPaper, authors: e.target.value})}
                            size="sm"
                          />
                        </Form.Group>
                      </Col>
                    </Row>
                    <Form.Group className="mb-3">
                      <Form.Label className="small">Abstract</Form.Label>
                      <Form.Control
                        as="textarea"
                        rows={2}
                        value={manualPaper.abstract}
                        onChange={(e) => setManualPaper({...manualPaper, abstract: e.target.value})}
                        size="sm"
                      />
                    </Form.Group>
                    <Form.Group className="mb-3">
                      <Form.Label className="small">URL (optional)</Form.Label>
                      <Form.Control
                        type="url"
                        value={manualPaper.url}
                        onChange={(e) => setManualPaper({...manualPaper, url: e.target.value})}
                        size="sm"
                      />
                    </Form.Group>
                    <div className="d-flex gap-2">
                      <Button variant="success" onClick={handleManualAdd} size="sm">
                        Add Paper
                      </Button>
                      <Button variant="outline-secondary" onClick={() => setManualMode(false)} size="sm">
                        Cancel
                      </Button>
                    </div>
                  </Card.Body>
                </Card>
              )}
            </Card.Body>
          </Card>
        </Col>
        </Row>
      </Container>
    </div>
  );
};

export default Home;

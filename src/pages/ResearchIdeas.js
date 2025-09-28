import React, { useState, useEffect } from 'react';
import { Container, Row, Col, Card, Button, Alert, Form, Modal, ProgressBar } from 'react-bootstrap';
import { FaLightbulb, FaPlus, FaHeart, FaShare, FaBrain, FaTrash, FaEdit } from 'react-icons/fa';
import { generateResearchIdeas } from '../services/api';

const ResearchIdeas = ({ readingList }) => {
  const [ideas, setIdeas] = useState([]);
  const [generatedIdeas, setGeneratedIdeas] = useState('');
  const [showGeneratedIdeas, setShowGeneratedIdeas] = useState(false);
  const [loading, setLoading] = useState(false);
  const [newIdea, setNewIdea] = useState({
    title: '',
    description: '',
    category: ''
  });
  const [editingIdea, setEditingIdea] = useState(null);

  const categories = [
    'Artificial Intelligence',
    'Machine Learning',
    'Computer Vision',
    'Natural Language Processing',
    'Data Science',
    'Cybersecurity',
    'Blockchain',
    'IoT',
    'Robotics',
    'Education Technology',
    'Healthcare Technology',
    'Environmental Science',
    'Other'
  ];

  useEffect(() => {
    // Load saved ideas from localStorage
    const savedIdeas = localStorage.getItem('profistant-research-ideas');
    if (savedIdeas) {
      setIdeas(JSON.parse(savedIdeas));
    }
  }, []);

  useEffect(() => {
    // Save ideas to localStorage
    localStorage.setItem('profistant-research-ideas', JSON.stringify(ideas));
  }, [ideas]);

  const generateAIResearchIdeas = async () => {
    if (!readingList || readingList.length === 0) {
      alert('Please add papers to your reading list first.');
      return;
    }

    setLoading(true);
    try {
      const response = await generateResearchIdeas(readingList);
      setGeneratedIdeas(response.ideas);
      setShowGeneratedIdeas(true);
    } catch (error) {
      alert('Failed to generate research ideas: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  const addGeneratedIdeasToList = () => {
    // Parse generated ideas and add them to the ideas list
    const ideaLines = generatedIdeas.split('\n').filter(line => 
      line.trim() && (line.trim().startsWith('•') || line.trim().startsWith('-') || line.trim().startsWith('*') || /^\d+\./.test(line.trim()))
    );
    
    const newIdeas = ideaLines.map((line, index) => ({
      id: Date.now() + index,
      title: line.trim().replace(/^[•\-*\d+\.\s]+/, '').substring(0, 100),
      description: line.trim().replace(/^[•\-*\d+\.\s]+/, ''),
      category: 'AI Generated',
      likes: 0,
      liked: false,
      aiGenerated: true
    }));

    setIdeas(prev => [...prev, ...newIdeas]);
    setShowGeneratedIdeas(false);
    alert(`Added ${newIdeas.length} AI-generated ideas to your list!`);
  };

  const toggleLike = (id) => {
    setIdeas(ideas.map(idea => 
      idea.id === id 
        ? { 
            ...idea, 
            liked: !idea.liked, 
            likes: idea.liked ? idea.likes - 1 : idea.likes + 1 
          }
        : idea
    ));
  };

  const addIdea = () => {
    if (newIdea.title && newIdea.description && newIdea.category) {
      setIdeas([...ideas, {
        id: Date.now(),
        ...newIdea,
        likes: 0,
        liked: false,
        aiGenerated: false
      }]);
      setNewIdea({ title: '', description: '', category: '' });
    }
  };

  const editIdea = (id) => {
    const idea = ideas.find(i => i.id === id);
    if (idea) {
      setEditingIdea(idea);
      setNewIdea({
        title: idea.title,
        description: idea.description,
        category: idea.category
      });
    }
  };

  const updateIdea = () => {
    if (editingIdea && newIdea.title && newIdea.description && newIdea.category) {
      setIdeas(ideas.map(idea => 
        idea.id === editingIdea.id 
          ? { ...idea, ...newIdea }
          : idea
      ));
      setEditingIdea(null);
      setNewIdea({ title: '', description: '', category: '' });
    }
  };

  const deleteIdea = (id) => {
    setIdeas(ideas.filter(idea => idea.id !== id));
  };

  return (
    <div className="main-content">
      <Container fluid>
        <div className="text-center mb-5">
          <h1 className="page-title">
            <FaLightbulb className="me-3" />
            Research Ideas & Gaps
          </h1>
          <p className="text-white-50 fs-5">
            Discover AI-powered research insights and manage your ideas
          </p>
        </div>

        {/* Reading List Check */}
        {(!readingList || readingList.length === 0) && (
          <Alert variant="warning" className="mb-4">
            <strong>No papers in reading list!</strong> Add papers from the Research page to use AI-powered idea generation.
          </Alert>
        )}

        <Row>
          <Col lg={8}>
            {/* AI-Powered Research Ideas Generation */}
            {(readingList && readingList.length > 0) && (
              <Card className="mb-4">
                <Card.Body>
                  <h5 className="card-title mb-3">
                    <FaBrain className="me-2 text-primary" />
                    AI-Powered Research Ideas
                  </h5>
                  <p className="text-muted mb-3">
                    Generate research gaps and project ideas based on your reading list using AI analysis.
                  </p>
                  
                  <Button 
                    variant="primary" 
                    onClick={generateAIResearchIdeas}
                    disabled={loading}
                    className="mb-3"
                  >
                    <FaBrain className="me-2" />
                    {loading ? 'Generating...' : 'Generate AI Ideas'}
                  </Button>

                  {loading && (
                    <div className="mb-3">
                      <ProgressBar animated now={100} />
                      <small className="text-muted">Analyzing your papers and generating research ideas...</small>
                    </div>
                  )}
                </Card.Body>
              </Card>
            )}

            {/* Manual Idea Entry */}
            <Card className="mb-4">
              <Card.Body>
                <h5 className="card-title mb-4">
                  <FaPlus className="me-2" />
                  Add Your Own Research Idea
                </h5>
                <Form>
                  <Row>
                    <Col md={6}>
                      <Form.Group className="mb-3">
                        <Form.Label>Research Idea Title</Form.Label>
                        <Form.Control
                          type="text"
                          placeholder="Enter your research idea title"
                          value={newIdea.title}
                          onChange={(e) => setNewIdea({...newIdea, title: e.target.value})}
                        />
                      </Form.Group>
                    </Col>
                    <Col md={6}>
                      <Form.Group className="mb-3">
                        <Form.Label>Category</Form.Label>
                        <Form.Select
                          value={newIdea.category}
                          onChange={(e) => setNewIdea({...newIdea, category: e.target.value})}
                        >
                          <option value="">Select a category</option>
                          {categories.map(category => (
                            <option key={category} value={category}>{category}</option>
                          ))}
                        </Form.Select>
                      </Form.Group>
                    </Col>
                  </Row>
                  <Form.Group className="mb-3">
                    <Form.Label>Description</Form.Label>
                    <Form.Control
                      as="textarea"
                      rows={3}
                      placeholder="Describe your research idea, methodology, or potential applications..."
                      value={newIdea.description}
                      onChange={(e) => setNewIdea({...newIdea, description: e.target.value})}
                    />
                  </Form.Group>
                  <div className="d-flex gap-2">
                    <Button 
                      variant="primary" 
                      onClick={editingIdea ? updateIdea : addIdea}
                      disabled={!newIdea.title || !newIdea.description || !newIdea.category}
                    >
                      <FaPlus className="me-2" />
                      {editingIdea ? 'Update Idea' : 'Add Idea'}
                    </Button>
                    {editingIdea && (
                      <Button 
                        variant="outline-secondary"
                        onClick={() => {
                          setEditingIdea(null);
                          setNewIdea({ title: '', description: '', category: '' });
                        }}
                      >
                        Cancel
                      </Button>
                    )}
                  </div>
                </Form>
              </Card.Body>
            </Card>

            {/* Ideas List */}
            <Card>
              <Card.Body>
                <h5 className="card-title mb-4">Your Research Ideas ({ideas.length})</h5>
                {ideas.length === 0 ? (
                  <Alert variant="info">
                    No ideas yet. Add your first research idea above or generate AI-powered ideas!
                  </Alert>
                ) : (
                  <div>
                    {ideas.map(idea => (
                      <Card key={idea.id} className="paper-card mb-3">
                        <Card.Body>
                          <div className="d-flex justify-content-between align-items-start mb-2">
                            <div className="flex-grow-1">
                              <h6 className="card-title mb-1">{idea.title}</h6>
                              <div className="d-flex gap-2 align-items-center mb-2">
                                <span className="badge bg-primary">{idea.category}</span>
                                {idea.aiGenerated && (
                                  <span className="badge bg-success">AI Generated</span>
                                )}
                              </div>
                            </div>
                            <div className="d-flex gap-1">
                              <Button
                                variant="outline-primary"
                                size="sm"
                                onClick={() => editIdea(idea.id)}
                              >
                                <FaEdit />
                              </Button>
                              <Button
                                variant="outline-danger"
                                size="sm"
                                onClick={() => deleteIdea(idea.id)}
                              >
                                <FaTrash />
                              </Button>
                            </div>
                          </div>
                          <p className="card-text mb-3">{idea.description}</p>
                          <div className="d-flex justify-content-between align-items-center">
                            <div className="d-flex gap-2">
                              <Button
                                variant={idea.liked ? "danger" : "outline-danger"}
                                size="sm"
                                onClick={() => toggleLike(idea.id)}
                              >
                                <FaHeart className="me-1" />
                                {idea.likes}
                              </Button>
                              <Button variant="outline-primary" size="sm">
                                <FaShare className="me-1" />
                                Share
                              </Button>
                            </div>
                            <small className="text-muted">
                              {idea.aiGenerated ? 'AI Generated' : 'Manual Entry'}
                            </small>
                          </div>
                        </Card.Body>
                      </Card>
                    ))}
                  </div>
                )}
              </Card.Body>
            </Card>
          </Col>

          <Col lg={4}>
            {/* Quick Stats */}
            <Card className="mb-4">
              <Card.Body>
                <h6 className="card-title">Quick Stats</h6>
                <div className="row text-center">
                  <div className="col-6">
                    <h4 className="text-primary">{ideas.length}</h4>
                    <small className="text-muted">Total Ideas</small>
                  </div>
                  <div className="col-6">
                    <h4 className="text-success">{ideas.filter(i => i.aiGenerated).length}</h4>
                    <small className="text-muted">AI Generated</small>
                  </div>
                </div>
              </Card.Body>
            </Card>

            {/* Categories Overview */}
            <Card>
              <Card.Body>
                <h6 className="card-title">Categories</h6>
                {categories.slice(0, 8).map((category, index) => {
                  const count = ideas.filter(idea => idea.category === category).length;
                  return (
                    <div key={index} className="d-flex justify-content-between align-items-center mb-1">
                      <span className="small">{category}</span>
                      <span className="badge bg-secondary">{count}</span>
                    </div>
                  );
                })}
              </Card.Body>
            </Card>
          </Col>
        </Row>

        {/* Generated Ideas Modal */}
        <Modal show={showGeneratedIdeas} onHide={() => setShowGeneratedIdeas(false)} size="lg">
          <Modal.Header closeButton>
            <Modal.Title>
              <FaBrain className="me-2" />
              AI-Generated Research Ideas
            </Modal.Title>
          </Modal.Header>
          <Modal.Body>
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
              {generatedIdeas}
            </div>
          </Modal.Body>
          <Modal.Footer>
            <Button variant="secondary" onClick={() => setShowGeneratedIdeas(false)}>
              Close
            </Button>
            <Button variant="success" onClick={addGeneratedIdeasToList}>
              <FaPlus className="me-2" />
              Add to Ideas List
            </Button>
          </Modal.Footer>
        </Modal>
      </Container>
    </div>
  );
};

export default ResearchIdeas;

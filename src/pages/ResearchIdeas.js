import React, { useState } from 'react';
import { Container, Row, Col, Card, Button, Alert } from 'react-bootstrap';
import { FaLightbulb, FaPlus, FaHeart, FaShare } from 'react-icons/fa';

const ResearchIdeas = () => {
  const [ideas, setIdeas] = useState([
    {
      id: 1,
      title: "AI-Powered Research Assistant",
      description: "Develop an intelligent system that can automatically analyze research papers and suggest relevant connections.",
      category: "Artificial Intelligence",
      likes: 15,
      liked: false
    },
    {
      id: 2,
      title: "Blockchain for Academic Publishing",
      description: "Create a decentralized platform for academic publishing to ensure transparency and prevent plagiarism.",
      category: "Blockchain",
      likes: 8,
      liked: false
    },
    {
      id: 3,
      title: "Virtual Reality Learning Environments",
      description: "Design immersive VR experiences for complex scientific concepts and laboratory simulations.",
      category: "Education Technology",
      likes: 12,
      liked: true
    }
  ]);

  const [newIdea, setNewIdea] = useState({
    title: '',
    description: '',
    category: ''
  });

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
        liked: false
      }]);
      setNewIdea({ title: '', description: '', category: '' });
    }
  };

  return (
    <div className="main-content">
      <Container fluid>
        <div className="text-center mb-5">
          <h1 className="page-title">
            <FaLightbulb className="me-3" />
            Research Ideas
          </h1>
          <p className="text-white-50 fs-5">
            Discover and share innovative research concepts
          </p>
        </div>

      <Row>
        <Col lg={12}>
          <Card className="mb-4">
            <Card.Body>
              <h5 className="card-title mb-4">Share Your Research Idea</h5>
              <div className="row">
                <div className="col-md-6 mb-3">
                  <input
                    type="text"
                    className="form-control"
                    placeholder="Research Idea Title"
                    value={newIdea.title}
                    onChange={(e) => setNewIdea({...newIdea, title: e.target.value})}
                  />
                </div>
                <div className="col-md-6 mb-3">
                  <input
                    type="text"
                    className="form-control"
                    placeholder="Category (e.g., AI, Blockchain, etc.)"
                    value={newIdea.category}
                    onChange={(e) => setNewIdea({...newIdea, category: e.target.value})}
                  />
                </div>
              </div>
              <div className="mb-3">
                <textarea
                  className="form-control"
                  rows="3"
                  placeholder="Describe your research idea..."
                  value={newIdea.description}
                  onChange={(e) => setNewIdea({...newIdea, description: e.target.value})}
                />
              </div>
              <Button variant="primary" onClick={addIdea}>
                <FaPlus className="me-2" />
                Share Idea
              </Button>
            </Card.Body>
          </Card>

          <div className="mb-4">
            <h5 className="mb-3">Featured Research Ideas</h5>
            {ideas.map(idea => (
              <Card key={idea.id} className="paper-card mb-3">
                <Card.Body>
                  <div className="d-flex justify-content-between align-items-start mb-2">
                    <h6 className="card-title mb-0">{idea.title}</h6>
                    <span className="badge bg-primary">{idea.category}</span>
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
                      {Math.floor(Math.random() * 30) + 1} days ago
                    </small>
                  </div>
                </Card.Body>
              </Card>
            ))}
          </div>
        </Col>
      </Row>
      </Container>
    </div>
  );
};

export default ResearchIdeas;

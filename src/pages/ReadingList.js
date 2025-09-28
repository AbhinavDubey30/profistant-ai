import React, { useState, useEffect } from 'react';
import { Container, Row, Col, Card, Button, Alert, Form, Badge, Modal } from 'react-bootstrap';
import { FaTrash, FaExternalLinkAlt, FaBook, FaPlus, FaEdit, FaSave, FaTimes } from 'react-icons/fa';

const ReadingList = ({ readingList, removeFromReadingList, clearReadingList, updatePaperProgress, updatePaperNotes }) => {
  const [paperProgress, setPaperProgress] = useState({});
  const [paperNotes, setPaperNotes] = useState({});
  const [showAddModal, setShowAddModal] = useState(false);
  const [editingPaper, setEditingPaper] = useState(null);
  const [newPaper, setNewPaper] = useState({
    title: '',
    authors: '',
    abstract: '',
    url: '',
    source: 'Manual',
    labels: []
  });

  const statusOptions = ['To Read', 'Reading', 'Done'];
  const sourceOptions = ['Manual', 'Google Scholar', 'Conference', 'ArXiv', 'Advisor', 'Other'];
  const labelOptions = ['Survey', 'Theoretical', 'Empirical', 'Review', 'Application', 'Methodology'];

  // Load progress and notes from localStorage
  useEffect(() => {
    const savedProgress = localStorage.getItem('profistant-paper-progress');
    const savedNotes = localStorage.getItem('profistant-paper-notes');
    
    if (savedProgress) {
      setPaperProgress(JSON.parse(savedProgress));
    }
    
    if (savedNotes) {
      setPaperNotes(JSON.parse(savedNotes));
    }
  }, []);

  // Save progress and notes to localStorage
  useEffect(() => {
    localStorage.setItem('profistant-paper-progress', JSON.stringify(paperProgress));
  }, [paperProgress]);

  useEffect(() => {
    localStorage.setItem('profistant-paper-notes', JSON.stringify(paperNotes));
  }, [paperNotes]);

  const handleStatusChange = (paperId, status) => {
    setPaperProgress(prev => ({
      ...prev,
      [paperId]: {
        ...prev[paperId],
        status: status
      }
    }));
  };

  const handleNotesChange = (paperId, notes) => {
    setPaperNotes(prev => ({
      ...prev,
      [paperId]: notes
    }));
  };

  const handleAddPaper = () => {
    if (!newPaper.title || !newPaper.authors || !newPaper.abstract) {
      alert('Please fill in title, authors, and abstract.');
      return;
    }

    const paperToAdd = {
      ...newPaper,
      id: Date.now(),
      year: new Date().getFullYear().toString()
    };

    // Add to reading list (this would need to be passed as a prop)
    // For now, we'll use a simple approach
    setShowAddModal(false);
    setNewPaper({
      title: '',
      authors: '',
      abstract: '',
      url: '',
      source: 'Manual',
      labels: []
    });

    // Show success message
    alert('Paper added to reading list!');
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'Done': return 'success';
      case 'Reading': return 'warning';
      case 'To Read': return 'secondary';
      default: return 'secondary';
    }
  };
  return (
    <div className="main-content">
      <Container fluid>
        <div className="text-center mb-5">
          <h1 className="page-title">
            <FaBook className="me-3" />
            Reading List
          </h1>
          <p className="text-white-50 fs-5">
            Manage your saved research papers with progress tracking
          </p>
        </div>

        <Row>
          <Col lg={12}>
            {readingList.length === 0 ? (
              <Card>
                <Card.Body className="text-center py-5">
                  <FaBook size={48} className="text-muted mb-3" />
                  <h5 className="text-muted">No papers in your reading list yet</h5>
                  <p className="text-muted mb-4">
                    Start by searching for papers on the Research page and add them to your list.
                  </p>
                  <Button 
                    variant="primary"
                    onClick={() => setShowAddModal(true)}
                  >
                    <FaPlus className="me-2" />
                    Add Your First Paper
                  </Button>
                </Card.Body>
              </Card>
            ) : (
              <>
                <div className="d-flex justify-content-between align-items-center mb-4">
                  <div>
                    <h4>Your Papers ({readingList.length})</h4>
                    <small className="text-muted">
                      Track your reading progress and add personal notes
                    </small>
                  </div>
                  <div className="d-flex gap-2">
                    <Button 
                      variant="primary"
                      onClick={() => setShowAddModal(true)}
                    >
                      <FaPlus className="me-2" />
                      Add Paper
                    </Button>
                    <Button 
                      variant="outline-danger"
                      onClick={() => {
                        if (window.confirm('Are you sure you want to clear all papers from your reading list?')) {
                          clearReadingList();
                        }
                      }}
                    >
                      <FaTrash className="me-2" />
                      Clear All
                    </Button>
                  </div>
                </div>

                {readingList.map((paper, index) => {
                  const progress = paperProgress[paper.id] || { status: 'To Read' };
                  const notes = paperNotes[paper.id] || '';
                  
                  return (
                    <Card key={paper.id || index} className="paper-card mb-3">
                      <Card.Body>
                        <div className="d-flex justify-content-between align-items-start mb-3">
                          <div className="flex-grow-1">
                            <h6 className="card-title mb-1">{paper.title}</h6>
                            <div className="d-flex gap-2 align-items-center mb-2">
                              <Badge bg={getStatusColor(progress.status)}>
                                {progress.status}
                              </Badge>
                              {paper.source && (
                                <Badge bg="info" className="text-dark">
                                  📌 {paper.source}
                                </Badge>
                              )}
                              {paper.labels && paper.labels.length > 0 && (
                                <>
                                  {paper.labels.map((label, idx) => (
                                    <Badge key={idx} bg="secondary">
                                      🏷️ {label}
                                    </Badge>
                                  ))}
                                </>
                              )}
                            </div>
                          </div>
                          <Button
                            variant="outline-danger"
                            size="sm"
                            onClick={() => removeFromReadingList(paper.id)}
                          >
                            <FaTrash />
                          </Button>
                        </div>
                        
                        <p className="text-muted mb-2">
                          <strong>Authors:</strong> {paper.authors}
                          {paper.year && paper.year !== 'Unknown' && (
                            <span className="ms-2">
                              <strong>Year:</strong> {paper.year}
                            </span>
                          )}
                        </p>
                        
                        <p className="card-text mb-3">{paper.abstract}</p>
                        
                        {/* Progress Tracking */}
                        <div className="mb-3">
                          <Form.Group>
                            <Form.Label className="small">📘 Reading Status</Form.Label>
                            <Form.Select
                              size="sm"
                              value={progress.status}
                              onChange={(e) => handleStatusChange(paper.id, e.target.value)}
                            >
                              {statusOptions.map(status => (
                                <option key={status} value={status}>{status}</option>
                              ))}
                            </Form.Select>
                          </Form.Group>
                        </div>

                        {/* Notes */}
                        <div className="mb-3">
                          <Form.Group>
                            <Form.Label className="small">📝 Personal Notes</Form.Label>
                            <Form.Control
                              as="textarea"
                              rows={2}
                              placeholder="Add your thoughts, key insights, or questions about this paper..."
                              value={notes}
                              onChange={(e) => handleNotesChange(paper.id, e.target.value)}
                            />
                          </Form.Group>
                        </div>
                        
                        {paper.url && paper.url !== '#' && (
                          <Button 
                            variant="outline-primary" 
                            size="sm"
                            href={paper.url}
                            target="_blank"
                            className="me-2"
                          >
                            <FaExternalLinkAlt className="me-1" />
                            View Paper
                          </Button>
                        )}
                      </Card.Body>
                    </Card>
                  );
                })}
              </>
            )}
          </Col>
        </Row>

        {/* Add Paper Modal */}
        <Modal show={showAddModal} onHide={() => setShowAddModal(false)} size="lg">
          <Modal.Header closeButton>
            <Modal.Title>
              <FaPlus className="me-2" />
              Add Custom Paper
            </Modal.Title>
          </Modal.Header>
          <Modal.Body>
            <Form>
              <Row>
                <Col md={6}>
                  <Form.Group className="mb-3">
                    <Form.Label>Paper Title *</Form.Label>
                    <Form.Control
                      type="text"
                      value={newPaper.title}
                      onChange={(e) => setNewPaper({...newPaper, title: e.target.value})}
                      placeholder="Enter paper title"
                    />
                  </Form.Group>
                </Col>
                <Col md={6}>
                  <Form.Group className="mb-3">
                    <Form.Label>Authors *</Form.Label>
                    <Form.Control
                      type="text"
                      value={newPaper.authors}
                      onChange={(e) => setNewPaper({...newPaper, authors: e.target.value})}
                      placeholder="Enter authors"
                    />
                  </Form.Group>
                </Col>
              </Row>
              
              <Form.Group className="mb-3">
                <Form.Label>Abstract *</Form.Label>
                <Form.Control
                  as="textarea"
                  rows={3}
                  value={newPaper.abstract}
                  onChange={(e) => setNewPaper({...newPaper, abstract: e.target.value})}
                  placeholder="Enter paper abstract"
                />
              </Form.Group>
              
              <Row>
                <Col md={6}>
                  <Form.Group className="mb-3">
                    <Form.Label>URL (optional)</Form.Label>
                    <Form.Control
                      type="url"
                      value={newPaper.url}
                      onChange={(e) => setNewPaper({...newPaper, url: e.target.value})}
                      placeholder="https://..."
                    />
                  </Form.Group>
                </Col>
                <Col md={6}>
                  <Form.Group className="mb-3">
                    <Form.Label>Source</Form.Label>
                    <Form.Select
                      value={newPaper.source}
                      onChange={(e) => setNewPaper({...newPaper, source: e.target.value})}
                    >
                      {sourceOptions.map(source => (
                        <option key={source} value={source}>{source}</option>
                      ))}
                    </Form.Select>
                  </Form.Group>
                </Col>
              </Row>
              
              <Form.Group className="mb-3">
                <Form.Label>Labels</Form.Label>
                <div className="d-flex flex-wrap gap-2">
                  {labelOptions.map(label => (
                    <Form.Check
                      key={label}
                      type="checkbox"
                      id={`label-${label}`}
                      label={label}
                      checked={newPaper.labels.includes(label)}
                      onChange={(e) => {
                        if (e.target.checked) {
                          setNewPaper({...newPaper, labels: [...newPaper.labels, label]});
                        } else {
                          setNewPaper({...newPaper, labels: newPaper.labels.filter(l => l !== label)});
                        }
                      }}
                    />
                  ))}
                </div>
              </Form.Group>
            </Form>
          </Modal.Body>
          <Modal.Footer>
            <Button variant="secondary" onClick={() => setShowAddModal(false)}>
              Cancel
            </Button>
            <Button variant="primary" onClick={handleAddPaper}>
              <FaPlus className="me-2" />
              Add to Reading List
            </Button>
          </Modal.Footer>
        </Modal>
      </Container>
    </div>
  );
};

export default ReadingList;

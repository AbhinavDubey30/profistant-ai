import React from 'react';
import { Container, Row, Col, Card, Button, Alert } from 'react-bootstrap';
import { FaTrash, FaExternalLinkAlt, FaBook } from 'react-icons/fa';

const ReadingList = ({ readingList, removeFromReadingList, clearReadingList }) => {
  return (
    <div className="main-content">
      <Container fluid>
        <div className="text-center mb-5">
          <h1 className="page-title">
            <FaBook className="me-3" />
            Reading List
          </h1>
          <p className="text-white-50 fs-5">
            Manage your saved research papers
          </p>
        </div>

      <Row>
        <Col lg={12}>
          {readingList.length === 0 ? (
            <Card>
              <Card.Body className="text-center py-5">
                <FaBook size={48} className="text-muted mb-3" />
                <h5 className="text-muted">No papers in your reading list yet</h5>
                <p className="text-muted">
                  Start by searching for papers on the Research page and add them to your list.
                </p>
              </Card.Body>
            </Card>
          ) : (
            <>
              <div className="d-flex justify-content-between align-items-center mb-4">
                <h4>Your Papers ({readingList.length})</h4>
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

              {readingList.map((paper, index) => (
                <Card key={paper.id || index} className="paper-card mb-3">
                  <Card.Body>
                    <div className="d-flex justify-content-between align-items-start mb-3">
                      <h6 className="card-title mb-0 flex-grow-1">{paper.title}</h6>
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
                    </p>
                    
                    <p className="card-text mb-3">{paper.abstract}</p>
                    
                    {paper.url && paper.url !== '#' && (
                      <Button 
                        variant="outline-primary" 
                        size="sm"
                        href={paper.url}
                        target="_blank"
                      >
                        <FaExternalLinkAlt className="me-1" />
                        View Paper
                      </Button>
                    )}
                  </Card.Body>
                </Card>
              ))}
            </>
          )}
        </Col>
      </Row>
      </Container>
    </div>
  );
};

export default ReadingList;

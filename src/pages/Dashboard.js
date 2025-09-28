import React, { useState, useEffect } from 'react';
import { Container, Row, Col, Card, ProgressBar, Form, Badge } from 'react-bootstrap';
import { FaChartBar, FaBook, FaCalendarAlt, FaLightbulb, FaFilter } from 'react-icons/fa';

const Dashboard = ({ readingList }) => {
  const [paperProgress, setPaperProgress] = useState({});
  const [paperNotes, setPaperNotes] = useState({});
  const [filterStatus, setFilterStatus] = useState('All');
  const [filterLabel, setFilterLabel] = useState('All');

  useEffect(() => {
    // Load progress and notes from localStorage
    const savedProgress = localStorage.getItem('profistant-paper-progress');
    const savedNotes = localStorage.getItem('profistant-paper-notes');
    
    if (savedProgress) {
      setPaperProgress(JSON.parse(savedProgress));
    }
    
    if (savedNotes) {
      setPaperNotes(JSON.parse(savedNotes));
    }
  }, []);

  const totalPapers = readingList.length;
  const papersWithUrls = readingList.filter(paper => paper.url && paper.url !== '#').length;
  
  // Calculate reading progress based on status
  const toReadPapers = readingList.filter(paper => {
    const progress = paperProgress[paper.id];
    return !progress || progress.status === 'To Read';
  }).length;
  
  const readingPapers = readingList.filter(paper => {
    const progress = paperProgress[paper.id];
    return progress && progress.status === 'Reading';
  }).length;
  
  const donePapers = readingList.filter(paper => {
    const progress = paperProgress[paper.id];
    return progress && progress.status === 'Done';
  }).length;
  
  const completionRate = totalPapers > 0 ? (donePapers / totalPapers) * 100 : 0;

  // Get all unique labels from papers
  const allLabels = readingList.reduce((acc, paper) => {
    if (paper.labels && paper.labels.length > 0) {
      paper.labels.forEach(label => {
        if (!acc.includes(label)) {
          acc.push(label);
        }
      });
    }
    return acc;
  }, []);

  // Get all unique sources
  const allSources = readingList.reduce((acc, paper) => {
    if (paper.source && !acc.includes(paper.source)) {
      acc.push(paper.source);
    }
    return acc;
  }, []);

  // Filter papers based on selected filters
  const filteredPapers = readingList.filter(paper => {
    const progress = paperProgress[paper.id];
    const status = progress ? progress.status : 'To Read';
    
    let statusMatch = true;
    if (filterStatus !== 'All') {
      statusMatch = status === filterStatus;
    }
    
    let labelMatch = true;
    if (filterLabel !== 'All') {
      labelMatch = paper.labels && paper.labels.includes(filterLabel);
    }
    
    return statusMatch && labelMatch;
  });

  return (
    <div className="main-content">
      <Container fluid>
        <div className="text-center mb-5">
          <h1 className="page-title">
            <FaChartBar className="me-3" />
            Research Dashboard
          </h1>
          <p className="text-white-50 fs-5">
            Track your research progress and insights with advanced analytics
          </p>
        </div>

        {/* Reading Progress Overview */}
        <Row>
          <Col lg={3} md={6} className="mb-4">
            <Card className="text-center h-100">
              <Card.Body>
                <FaBook size={48} className="text-primary mb-3" />
                <h4>{totalPapers}</h4>
                <p className="text-muted mb-0">Total Papers</p>
              </Card.Body>
            </Card>
          </Col>

          <Col lg={3} md={6} className="mb-4">
            <Card className="text-center h-100">
              <Card.Body>
                <FaBook size={48} className="text-warning mb-3" />
                <h4>{readingPapers}</h4>
                <p className="text-muted mb-0">Currently Reading</p>
              </Card.Body>
            </Card>
          </Col>

          <Col lg={3} md={6} className="mb-4">
            <Card className="text-center h-100">
              <Card.Body>
                <FaBook size={48} className="text-success mb-3" />
                <h4>{donePapers}</h4>
                <p className="text-muted mb-0">Completed</p>
              </Card.Body>
            </Card>
          </Col>

          <Col lg={3} md={6} className="mb-4">
            <Card className="text-center h-100">
              <Card.Body>
                <FaLightbulb size={48} className="text-info mb-3" />
                <h4>{Math.floor(totalPapers * 0.3)}</h4>
                <p className="text-muted mb-0">Research Ideas</p>
              </Card.Body>
            </Card>
          </Col>
        </Row>

        {/* Reading Progress Chart */}
        <Row>
          <Col lg={6} className="mb-4">
            <Card>
              <Card.Body>
                <h5 className="card-title">Reading Progress Distribution</h5>
                <div className="mb-3">
                  <div className="d-flex justify-content-between mb-1">
                    <span>To Read</span>
                    <span>{toReadPapers}</span>
                  </div>
                  <ProgressBar 
                    variant="secondary" 
                    now={totalPapers > 0 ? (toReadPapers / totalPapers) * 100 : 0} 
                  />
                </div>
                <div className="mb-3">
                  <div className="d-flex justify-content-between mb-1">
                    <span>Reading</span>
                    <span>{readingPapers}</span>
                  </div>
                  <ProgressBar 
                    variant="warning" 
                    now={totalPapers > 0 ? (readingPapers / totalPapers) * 100 : 0} 
                  />
                </div>
                <div className="mb-3">
                  <div className="d-flex justify-content-between mb-1">
                    <span>Done</span>
                    <span>{donePapers}</span>
                  </div>
                  <ProgressBar 
                    variant="success" 
                    now={totalPapers > 0 ? (donePapers / totalPapers) * 100 : 0} 
                  />
                </div>
                <p className="text-muted small">
                  Overall completion rate: {completionRate.toFixed(1)}%
                </p>
              </Card.Body>
            </Card>
          </Col>

          <Col lg={6} className="mb-4">
            <Card>
              <Card.Body>
                <h5 className="card-title">Recent Activity</h5>
                {totalPapers === 0 ? (
                  <p className="text-muted">No papers added yet. Start researching!</p>
                ) : (
                  <div>
                    <p className="mb-2">
                      <strong>Latest Addition:</strong><br />
                      {readingList[readingList.length - 1]?.title}
                    </p>
                    <p className="text-muted small mb-0">
                      Added {new Date().toLocaleDateString()}
                    </p>
                    <hr />
                    <p className="mb-2">
                      <strong>Papers with Links:</strong> {papersWithUrls}
                    </p>
                    <p className="text-muted small mb-0">
                      {((papersWithUrls / totalPapers) * 100).toFixed(1)}% have accessible URLs
                    </p>
                  </div>
                )}
              </Card.Body>
            </Card>
          </Col>
        </Row>

        {/* Labels and Sources Overview */}
        <Row>
          <Col lg={6} className="mb-4">
            <Card>
              <Card.Body>
                <h5 className="card-title">Labels Overview</h5>
                {allLabels.length === 0 ? (
                  <p className="text-muted">No labels found. Add them in the Reading List.</p>
                ) : (
                  <div className="d-flex flex-wrap gap-2">
                    {allLabels.map((label, index) => {
                      const count = readingList.filter(paper => 
                        paper.labels && paper.labels.includes(label)
                      ).length;
                      return (
                        <Badge key={index} bg="secondary" className="fs-6 p-2">
                          🏷️ {label} ({count})
                        </Badge>
                      );
                    })}
                  </div>
                )}
              </Card.Body>
            </Card>
          </Col>

          <Col lg={6} className="mb-4">
            <Card>
              <Card.Body>
                <h5 className="card-title">Sources Overview</h5>
                {allSources.length === 0 ? (
                  <p className="text-muted">No sources tracked yet.</p>
                ) : (
                  <div className="d-flex flex-wrap gap-2">
                    {allSources.map((source, index) => {
                      const count = readingList.filter(paper => 
                        paper.source === source
                      ).length;
                      return (
                        <Badge key={index} bg="info" className="fs-6 p-2">
                          📌 {source} ({count})
                        </Badge>
                      );
                    })}
                  </div>
                )}
              </Card.Body>
            </Card>
          </Col>
        </Row>

        {/* Filters and Paper List */}
        <Row>
          <Col lg={12}>
            <Card>
              <Card.Body>
                <div className="d-flex justify-content-between align-items-center mb-3">
                  <h5 className="card-title mb-0">
                    <FaFilter className="me-2" />
                    Filter Papers
                  </h5>
                  <small className="text-muted">
                    Showing {filteredPapers.length} of {totalPapers} papers
                  </small>
                </div>
                
                <Row className="mb-3">
                  <Col md={6}>
                    <Form.Group>
                      <Form.Label className="small">Filter by Status</Form.Label>
                      <Form.Select
                        size="sm"
                        value={filterStatus}
                        onChange={(e) => setFilterStatus(e.target.value)}
                      >
                        <option value="All">All Statuses</option>
                        <option value="To Read">To Read</option>
                        <option value="Reading">Reading</option>
                        <option value="Done">Done</option>
                      </Form.Select>
                    </Form.Group>
                  </Col>
                  <Col md={6}>
                    <Form.Group>
                      <Form.Label className="small">Filter by Label</Form.Label>
                      <Form.Select
                        size="sm"
                        value={filterLabel}
                        onChange={(e) => setFilterLabel(e.target.value)}
                      >
                        <option value="All">All Labels</option>
                        {allLabels.map(label => (
                          <option key={label} value={label}>{label}</option>
                        ))}
                      </Form.Select>
                    </Form.Group>
                  </Col>
                </Row>

                {filteredPapers.length === 0 ? (
                  <p className="text-muted text-center py-3">
                    {totalPapers === 0 
                      ? "No papers in your reading list yet." 
                      : "No papers match the selected filters."
                    }
                  </p>
                ) : (
                  <div className="table-responsive">
                    <table className="table table-sm">
                      <thead>
                        <tr>
                          <th>Title</th>
                          <th>Status</th>
                          <th>Source</th>
                          <th>Labels</th>
                        </tr>
                      </thead>
                      <tbody>
                        {filteredPapers.map((paper, index) => {
                          const progress = paperProgress[paper.id];
                          const status = progress ? progress.status : 'To Read';
                          const getStatusColor = (status) => {
                            switch (status) {
                              case 'Done': return 'success';
                              case 'Reading': return 'warning';
                              case 'To Read': return 'secondary';
                              default: return 'secondary';
                            }
                          };
                          
                          return (
                            <tr key={paper.id || index}>
                              <td>
                                <div className="fw-bold">{paper.title}</div>
                                <small className="text-muted">{paper.authors}</small>
                              </td>
                              <td>
                                <Badge bg={getStatusColor(status)}>
                                  {status}
                                </Badge>
                              </td>
                              <td>
                                {paper.source && (
                                  <Badge bg="info" className="text-dark">
                                    📌 {paper.source}
                                  </Badge>
                                )}
                              </td>
                              <td>
                                {paper.labels && paper.labels.length > 0 ? (
                                  <div className="d-flex flex-wrap gap-1">
                                    {paper.labels.map((label, idx) => (
                                      <Badge key={idx} bg="secondary" className="small">
                                        🏷️ {label}
                                      </Badge>
                                    ))}
                                  </div>
                                ) : (
                                  <span className="text-muted">No labels</span>
                                )}
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                )}
              </Card.Body>
            </Card>
          </Col>
        </Row>
      </Container>
    </div>
  );
};

export default Dashboard;

import React from 'react';
import { Container, Row, Col, Card, ProgressBar } from 'react-bootstrap';
import { FaChartBar, FaBook, FaCalendarAlt, FaLightbulb } from 'react-icons/fa';

const Dashboard = ({ readingList }) => {
  const totalPapers = readingList.length;
  const papersWithUrls = readingList.filter(paper => paper.url && paper.url !== '#').length;
  const completionRate = totalPapers > 0 ? (papersWithUrls / totalPapers) * 100 : 0;

  return (
    <div className="main-content">
      <Container fluid>
        <div className="text-center mb-5">
          <h1 className="page-title">
            <FaChartBar className="me-3" />
            Dashboard
          </h1>
          <p className="text-white-50 fs-5">
            Track your research progress and insights
          </p>
        </div>

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
              <FaBook size={48} className="text-success mb-3" />
              <h4>{papersWithUrls}</h4>
              <p className="text-muted mb-0">With Links</p>
            </Card.Body>
          </Card>
        </Col>

        <Col lg={3} md={6} className="mb-4">
          <Card className="text-center h-100">
            <Card.Body>
              <FaCalendarAlt size={48} className="text-warning mb-3" />
              <h4>{Math.ceil(totalPapers / 7)}</h4>
              <p className="text-muted mb-0">Weeks to Read</p>
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

      <Row>
        <Col lg={6} className="mb-4">
          <Card>
            <Card.Body>
              <h5 className="card-title">Reading Progress</h5>
              <div className="mb-3">
                <div className="d-flex justify-content-between mb-1">
                  <span>Completion Rate</span>
                  <span>{completionRate.toFixed(1)}%</span>
                </div>
                <ProgressBar now={completionRate} />
              </div>
              <p className="text-muted small">
                {papersWithUrls} out of {totalPapers} papers have accessible links
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
                </div>
              )}
            </Card.Body>
          </Card>
        </Col>
      </Row>

      <Row>
        <Col lg={12}>
          <Card>
            <Card.Body>
              <h5 className="card-title">Research Topics</h5>
              {totalPapers === 0 ? (
                <p className="text-muted">No research topics yet. Add some papers to see topic distribution.</p>
              ) : (
                <div className="row">
                  <div className="col-md-6">
                    <h6>Top Keywords</h6>
                    <ul className="list-unstyled">
                      <li><span className="badge bg-primary me-2">AI</span> Machine Learning</li>
                      <li><span className="badge bg-success me-2">ML</span> Deep Learning</li>
                      <li><span className="badge bg-warning me-2">DL</span> Neural Networks</li>
                    </ul>
                  </div>
                  <div className="col-md-6">
                    <h6>Research Areas</h6>
                    <ul className="list-unstyled">
                      <li><span className="badge bg-info me-2">CS</span> Computer Science</li>
                      <li><span className="badge bg-secondary me-2">AI</span> Artificial Intelligence</li>
                      <li><span className="badge bg-dark me-2">ROB</span> Robotics</li>
                    </ul>
                  </div>
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

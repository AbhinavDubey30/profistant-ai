import React from 'react';
import { Container, Row, Col, Card, Button } from 'react-bootstrap';
import { FaGraduationCap, FaSearch, FaBook, FaChartBar, FaCalendarAlt, FaLightbulb, FaGithub, FaArrowRight } from 'react-icons/fa';
import { Link } from 'react-router-dom';

const Landing = () => {
  const features = [
    {
      icon: FaSearch,
      title: "Smart Paper Search",
      description: "Search Google Scholar with advanced retry mechanisms and error handling",
      color: "primary"
    },
    {
      icon: FaBook,
      title: "Reading List Management",
      description: "Save and organize your research papers with local storage",
      color: "success"
    },
    {
      icon: FaChartBar,
      title: "Research Dashboard",
      description: "Track your research progress with analytics and insights",
      color: "info"
    },
    {
      icon: FaCalendarAlt,
      title: "Research Planner",
      description: "Plan and track your research activities and tasks",
      color: "warning"
    },
    {
      icon: FaLightbulb,
      title: "Research Ideas",
      description: "Discover and share innovative research concepts",
      color: "secondary"
    }
  ];

  return (
    <div className="landing-page">
      {/* Hero Section */}
      <div className="hero-section">
        <Container>
          <Row className="align-items-center min-vh-100">
            <Col lg={6}>
              <div className="hero-content">
                <h1 className="hero-title">
                  <FaGraduationCap className="me-3" />
                  Profistant AI
                </h1>
                <h2 className="hero-subtitle">
                  Your Research Kick-Starter
                </h2>
                <p className="hero-description">
                  An AI-powered research assistant designed to revolutionize your academic journey. 
                  Search papers, manage reading lists, plan research activities, and discover new ideas 
                  all in one beautiful, modern interface.
                </p>
                <div className="hero-buttons">
                  <Button 
                    as={Link} 
                    to="/research" 
                    variant="primary" 
                    size="lg"
                    className="me-3"
                  >
                    Start Researching
                    <FaArrowRight className="ms-2" />
                  </Button>
                  <Button 
                    as={Link} 
                    to="/dashboard" 
                    variant="outline-light" 
                    size="lg"
                  >
                    View Dashboard
                  </Button>
                </div>
              </div>
            </Col>
            <Col lg={6}>
              <div className="hero-image">
                <div className="floating-cards">
                  <Card className="floating-card card-1">
                    <Card.Body>
                      <FaSearch className="text-primary mb-2" />
                      <h6>Smart Search</h6>
                      <small>AI-powered paper discovery</small>
                    </Card.Body>
                  </Card>
                  <Card className="floating-card card-2">
                    <Card.Body>
                      <FaBook className="text-success mb-2" />
                      <h6>Reading List</h6>
                      <small>Organize your research</small>
                    </Card.Body>
                  </Card>
                  <Card className="floating-card card-3">
                    <Card.Body>
                      <FaChartBar className="text-info mb-2" />
                      <h6>Analytics</h6>
                      <small>Track your progress</small>
                    </Card.Body>
                  </Card>
                </div>
              </div>
            </Col>
          </Row>
        </Container>
      </div>

      {/* Features Section */}
      <div className="features-section">
        <Container>
          <Row>
            <Col lg={12} className="text-center mb-5">
              <h2 className="section-title">Powerful Features</h2>
              <p className="section-description">
                Everything you need for efficient academic research
              </p>
            </Col>
          </Row>
          <Row>
            {features.map((feature, index) => {
              const IconComponent = feature.icon;
              return (
                <Col lg={4} md={6} className="mb-4" key={index}>
                  <Card className="feature-card h-100">
                    <Card.Body className="text-center">
                      <div className={`feature-icon bg-${feature.color}`}>
                        <IconComponent />
                      </div>
                      <h5 className="feature-title">{feature.title}</h5>
                      <p className="feature-description">{feature.description}</p>
                    </Card.Body>
                  </Card>
                </Col>
              );
            })}
          </Row>
        </Container>
      </div>

      {/* CTA Section */}
      <div className="cta-section">
        <Container>
          <Row>
            <Col lg={12} className="text-center">
              <h2 className="cta-title">Ready to Transform Your Research?</h2>
              <p className="cta-description">
                Join researchers worldwide who are using Profistant AI to accelerate their academic journey
              </p>
              <Button 
                as={Link} 
                to="/research" 
                variant="primary" 
                size="lg"
                className="cta-button"
              >
                Get Started Now
                <FaArrowRight className="ms-2" />
              </Button>
            </Col>
          </Row>
        </Container>
      </div>

      {/* Footer */}
      <footer className="footer">
        <Container>
          <Row className="align-items-center">
            <Col lg={6}>
              <div className="footer-content">
                <h5 className="footer-title">
                  <FaGraduationCap className="me-2" />
                  Profistant AI
                </h5>
                <p className="footer-description">
                  Empowering researchers with AI-driven tools for academic excellence
                </p>
              </div>
            </Col>
            <Col lg={6} className="text-lg-end">
              <div className="footer-info">
                <p className="developer-info">
                  <strong>Developed by:</strong> Abhinav Sudhakar Dubey
                </p>
                <p className="github-link">
                  <FaGithub className="me-2" />
                  <a 
                    href="https://github.com/AbhinavDubey30" 
                    target="_blank" 
                    rel="noopener noreferrer"
                    className="github-link-text"
                  >
                    @AbhinavDubey30
                  </a>
                </p>
                <p className="copyright">
                  © 2025 Profistant AI. All rights reserved.
                </p>
              </div>
            </Col>
          </Row>
        </Container>
      </footer>
    </div>
  );
};

export default Landing;

import React from 'react';
import { Navbar as BSNavbar, Nav, Container, Button } from 'react-bootstrap';
import { Link, useLocation } from 'react-router-dom';
import { FaGraduationCap, FaBook, FaCalendarAlt, FaChartBar, FaLightbulb, FaBars, FaTimes } from 'react-icons/fa';

const Navbar = ({ sidebarOpen, toggleSidebar, readingList }) => {
  const location = useLocation();

  const menuItems = [
    { path: '/research', icon: FaBook, label: 'Research' },
    { path: '/reading-list', icon: FaBook, label: 'Reading List' },
    { path: '/planner', icon: FaCalendarAlt, label: 'Planner' },
    { path: '/dashboard', icon: FaChartBar, label: 'Dashboard' },
    { path: '/research-ideas', icon: FaLightbulb, label: 'Research Ideas' }
  ];

  return (
    <BSNavbar bg="dark" variant="dark" expand="lg" className="top-navbar">
      <Container fluid>
        <div className="d-flex align-items-center">
          <Button
            variant="outline-light"
            size="sm"
            onClick={toggleSidebar}
            className="me-3 sidebar-toggle-btn"
          >
            {sidebarOpen ? <FaTimes /> : <FaBars />}
          </Button>
          <BSNavbar.Brand as={Link} to="/research" className="fw-bold">
            <FaGraduationCap className="me-2" />
            Profistant AI
          </BSNavbar.Brand>
        </div>
        
        <BSNavbar.Toggle aria-controls="basic-navbar-nav" />
        <BSNavbar.Collapse id="basic-navbar-nav">
          <Nav className="ms-auto">
            {menuItems.map((item) => {
              const IconComponent = item.icon;
              const isActive = location.pathname === item.path;
              
              return (
                <Nav.Link 
                  key={item.path}
                  as={Link} 
                  to={item.path} 
                  className={isActive ? 'active' : ''}
                >
                  <IconComponent className="me-1" />
                  {item.label}
                  {item.path === '/reading-list' && readingList.length > 0 && (
                    <span className="badge bg-primary ms-1">{readingList.length}</span>
                  )}
                </Nav.Link>
              );
            })}
          </Nav>
        </BSNavbar.Collapse>
      </Container>
    </BSNavbar>
  );
};

export default Navbar;
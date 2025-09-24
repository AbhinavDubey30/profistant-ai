import React from 'react';
import { Nav, Button } from 'react-bootstrap';
import { Link, useLocation } from 'react-router-dom';
import { FaGraduationCap, FaBook, FaCalendarAlt, FaChartBar, FaLightbulb, FaTrash, FaTimes } from 'react-icons/fa';

const Sidebar = ({ readingList, clearReadingList, sidebarOpen, toggleSidebar }) => {
  const location = useLocation();

  const menuItems = [
    { path: '/research', icon: FaBook, label: 'Research', color: 'primary' },
    { path: '/reading-list', icon: FaBook, label: 'Reading List', color: 'success' },
    { path: '/planner', icon: FaCalendarAlt, label: 'Planner', color: 'warning' },
    { path: '/dashboard', icon: FaChartBar, label: 'Dashboard', color: 'info' },
    { path: '/research-ideas', icon: FaLightbulb, label: 'Research Ideas', color: 'secondary' }
  ];

  return (
    <div className={`sidebar ${sidebarOpen ? 'sidebar-open' : 'sidebar-closed'}`}>
      <div className="sidebar-header">
        <div className="brand">
          <FaGraduationCap className="brand-icon" />
          <span className="brand-text">Profistant AI</span>
        </div>
        <Button
          variant="outline-light"
          size="sm"
          onClick={toggleSidebar}
          className="sidebar-close-btn"
        >
          <FaTimes />
        </Button>
      </div>

      <div className="sidebar-menu">
        <Nav className="flex-column">
          {menuItems.map((item) => {
            const IconComponent = item.icon;
            const isActive = location.pathname === item.path;
            
            return (
              <Nav.Item key={item.path}>
                <Nav.Link 
                  as={Link} 
                  to={item.path}
                  className={`menu-item ${isActive ? 'active' : ''}`}
                >
                  <IconComponent className="menu-icon" />
                  <span className="menu-label">{item.label}</span>
                  {item.path === '/reading-list' && readingList.length > 0 && (
                    <span className="badge bg-primary ms-auto">{readingList.length}</span>
                  )}
                </Nav.Link>
              </Nav.Item>
            );
          })}
        </Nav>
      </div>

      <div className="sidebar-footer">
        <div className="reading-list-summary">
          <h6>Reading List</h6>
          <p className="text-muted mb-2">
            Papers saved: <strong>{readingList.length}</strong>
          </p>
          {readingList.length > 0 && (
            <button 
              className="btn btn-outline-danger btn-sm w-100"
              onClick={() => {
                if (window.confirm('Clear all papers from reading list?')) {
                  clearReadingList();
                }
              }}
            >
              <FaTrash className="me-1" />
              Clear List
            </button>
          )}
        </div>
      </div>
    </div>
  );
};

export default Sidebar;

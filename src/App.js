import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import Navbar from './components/Navbar';
import Landing from './pages/Landing';
import Home from './pages/Home';
import ReadingList from './pages/ReadingList';
import Planner from './pages/Planner';
import Dashboard from './pages/Dashboard';
import ResearchIdeas from './pages/ResearchIdeas';
import './App.css';

function App() {
  const [readingList, setReadingList] = useState([]);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [settings, setSettings] = useState({
    timeout: 15,
    useProxy: false,
    proxyHost: '',
    proxyPort: ''
  });

  // Load data from localStorage on component mount
  useEffect(() => {
    const savedReadingList = localStorage.getItem('profistant-reading-list');
    const savedSettings = localStorage.getItem('profistant-settings');
    const savedSidebarState = localStorage.getItem('profistant-sidebar-open');
    
    if (savedReadingList) {
      setReadingList(JSON.parse(savedReadingList));
    }
    
    if (savedSettings) {
      setSettings(JSON.parse(savedSettings));
    }
    
    if (savedSidebarState !== null) {
      setSidebarOpen(JSON.parse(savedSidebarState));
    }
  }, []);

  // Save data to localStorage whenever it changes
  useEffect(() => {
    localStorage.setItem('profistant-reading-list', JSON.stringify(readingList));
  }, [readingList]);

  useEffect(() => {
    localStorage.setItem('profistant-settings', JSON.stringify(settings));
  }, [settings]);

  useEffect(() => {
    localStorage.setItem('profistant-sidebar-open', JSON.stringify(sidebarOpen));
  }, [sidebarOpen]);

  const addToReadingList = (paper) => {
    setReadingList(prev => [...prev, { ...paper, id: Date.now() }]);
  };

  const removeFromReadingList = (id) => {
    setReadingList(prev => prev.filter(paper => paper.id !== id));
  };

  const clearReadingList = () => {
    setReadingList([]);
  };

  const toggleSidebar = () => {
    setSidebarOpen(!sidebarOpen);
  };

  return (
    <Router>
      <div className="App">
        <Routes>
          <Route path="/" element={<Landing />} />
          <Route 
            path="/research" 
            element={
              <div className="app-container">
                <Navbar 
                  sidebarOpen={sidebarOpen} 
                  toggleSidebar={toggleSidebar} 
                  readingList={readingList} 
                />
                <Sidebar 
                  readingList={readingList} 
                  clearReadingList={clearReadingList}
                  sidebarOpen={sidebarOpen}
                  toggleSidebar={toggleSidebar}
                />
                <div className={`main-content ${sidebarOpen ? 'sidebar-open' : 'sidebar-closed'}`}>
                  <Home 
                    readingList={readingList}
                    addToReadingList={addToReadingList}
                    settings={settings}
                    setSettings={setSettings}
                  />
                </div>
              </div>
            } 
          />
          <Route 
            path="/reading-list" 
            element={
              <div className="app-container">
                <Navbar 
                  sidebarOpen={sidebarOpen} 
                  toggleSidebar={toggleSidebar} 
                  readingList={readingList} 
                />
                <Sidebar 
                  readingList={readingList} 
                  clearReadingList={clearReadingList}
                  sidebarOpen={sidebarOpen}
                  toggleSidebar={toggleSidebar}
                />
                <div className={`main-content ${sidebarOpen ? 'sidebar-open' : 'sidebar-closed'}`}>
                  <ReadingList 
                    readingList={readingList}
                    removeFromReadingList={removeFromReadingList}
                    clearReadingList={clearReadingList}
                  />
                </div>
              </div>
            } 
          />
          <Route 
            path="/planner" 
            element={
              <div className="app-container">
                <Navbar 
                  sidebarOpen={sidebarOpen} 
                  toggleSidebar={toggleSidebar} 
                  readingList={readingList} 
                />
                <Sidebar 
                  readingList={readingList} 
                  clearReadingList={clearReadingList}
                  sidebarOpen={sidebarOpen}
                  toggleSidebar={toggleSidebar}
                />
                <div className={`main-content ${sidebarOpen ? 'sidebar-open' : 'sidebar-closed'}`}>
                  <Planner readingList={readingList} />
                </div>
              </div>
            } 
          />
          <Route 
            path="/dashboard" 
            element={
              <div className="app-container">
                <Navbar 
                  sidebarOpen={sidebarOpen} 
                  toggleSidebar={toggleSidebar} 
                  readingList={readingList} 
                />
                <Sidebar 
                  readingList={readingList} 
                  clearReadingList={clearReadingList}
                  sidebarOpen={sidebarOpen}
                  toggleSidebar={toggleSidebar}
                />
                <div className={`main-content ${sidebarOpen ? 'sidebar-open' : 'sidebar-closed'}`}>
                  <Dashboard readingList={readingList} />
                </div>
              </div>
            } 
          />
          <Route 
            path="/research-ideas" 
            element={
              <div className="app-container">
                <Navbar 
                  sidebarOpen={sidebarOpen} 
                  toggleSidebar={toggleSidebar} 
                  readingList={readingList} 
                />
                <Sidebar 
                  readingList={readingList} 
                  clearReadingList={clearReadingList}
                  sidebarOpen={sidebarOpen}
                  toggleSidebar={toggleSidebar}
                />
                <div className={`main-content ${sidebarOpen ? 'sidebar-open' : 'sidebar-closed'}`}>
                  <ResearchIdeas readingList={readingList} />
                </div>
              </div>
            } 
          />
        </Routes>
      </div>
    </Router>
  );
}

export default App;

import React, { useState, useEffect } from 'react';
import { Container, Row, Col, Card, Form, Button, Alert, Modal, ProgressBar } from 'react-bootstrap';
import { FaCalendarAlt, FaPlus, FaTrash, FaCheck, FaDownload, FaBrain, FaClock } from 'react-icons/fa';
import { generateCalendar, generateResearchPlan } from '../services/api';

const Planner = ({ readingList }) => {
  const [tasks, setTasks] = useState([]);
  const [newTask, setNewTask] = useState('');
  const [showSettings, setShowSettings] = useState(false);
  const [showPlan, setShowPlan] = useState(false);
  const [researchPlan, setResearchPlan] = useState('');
  const [calendarData, setCalendarData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [settings, setSettings] = useState({
    totalWeeks: 4,
    selectedDays: ['Monday', 'Wednesday', 'Friday'],
    timeRange: [17, 19]
  });

  const daysOfWeek = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];

  useEffect(() => {
    // Load tasks from localStorage
    const savedTasks = localStorage.getItem('profistant-tasks');
    if (savedTasks) {
      setTasks(JSON.parse(savedTasks));
    }
    
    // Load settings from localStorage
    const savedSettings = localStorage.getItem('profistant-planner-settings');
    if (savedSettings) {
      setSettings(JSON.parse(savedSettings));
    }
  }, []);

  useEffect(() => {
    // Save tasks to localStorage
    localStorage.setItem('profistant-tasks', JSON.stringify(tasks));
  }, [tasks]);

  useEffect(() => {
    // Save settings to localStorage
    localStorage.setItem('profistant-planner-settings', JSON.stringify(settings));
  }, [settings]);

  const addTask = () => {
    if (newTask.trim()) {
      setTasks([...tasks, {
        id: Date.now(),
        text: newTask,
        completed: false,
        date: new Date().toISOString().split('T')[0]
      }]);
      setNewTask('');
    }
  };

  const toggleTask = (id) => {
    setTasks(tasks.map(task => 
      task.id === id ? { ...task, completed: !task.completed } : task
    ));
  };

  const deleteTask = (id) => {
    setTasks(tasks.filter(task => task.id !== id));
  };

  const generateAIResearchPlan = async () => {
    if (!readingList || readingList.length === 0) {
      alert('Please add papers to your reading list first.');
      return;
    }

    setLoading(true);
    try {
      const response = await generateResearchPlan(
        readingList,
        settings.totalWeeks,
        settings.selectedDays,
        settings.timeRange
      );
      setResearchPlan(response.plan);
      setShowPlan(true);
    } catch (error) {
      alert('Failed to generate research plan: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  const generateCalendarFile = async () => {
    if (!readingList || readingList.length === 0) {
      alert('Please add papers to your reading list first.');
      return;
    }

    setLoading(true);
    try {
      const response = await generateCalendar(
        readingList,
        settings.totalWeeks,
        settings.selectedDays,
        settings.timeRange
      );
      
      // Create and download the calendar file
      const blob = new Blob([response.calendar_content], { type: 'text/calendar' });
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = response.filename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
      
      alert('Calendar downloaded successfully!');
    } catch (error) {
      alert('Failed to generate calendar: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  const completedTasks = tasks.filter(task => task.completed).length;
  const totalTasks = tasks.length;

  return (
    <div className="main-content">
      <Container fluid>
        <div className="text-center mb-5">
          <h1 className="page-title">
            <FaCalendarAlt className="me-3" />
            Research Planner
          </h1>
          <p className="text-white-50 fs-5">
            Plan and track your research activities with AI-powered scheduling
          </p>
        </div>

        {/* Reading List Check */}
        {(!readingList || readingList.length === 0) && (
          <Alert variant="warning" className="mb-4">
            <strong>No papers in reading list!</strong> Add papers from the Research page to use AI-powered planning features.
          </Alert>
        )}

        <Row>
          <Col lg={8}>
            {/* AI-Powered Research Planning */}
            {(readingList && readingList.length > 0) && (
              <Card className="mb-4">
                <Card.Body>
                  <h5 className="card-title mb-3">
                    <FaBrain className="me-2 text-primary" />
                    AI-Powered Research Planning
                  </h5>
                  <p className="text-muted mb-3">
                    Generate personalized research schedules based on your reading list and availability.
                  </p>
                  
                  <div className="d-flex gap-2 mb-3">
                    <Button 
                      variant="primary" 
                      onClick={generateAIResearchPlan}
                      disabled={loading}
                    >
                      <FaBrain className="me-2" />
                      {loading ? 'Generating...' : 'Generate AI Plan'}
                    </Button>
                    <Button 
                      variant="success" 
                      onClick={generateCalendarFile}
                      disabled={loading}
                    >
                      <FaDownload className="me-2" />
                      Download Calendar
                    </Button>
                    <Button 
                      variant="outline-secondary"
                      onClick={() => setShowSettings(true)}
                    >
                      <FaClock className="me-2" />
                      Settings
                    </Button>
                  </div>

                  {loading && (
                    <div className="mb-3">
                      <ProgressBar animated now={100} />
                      <small className="text-muted">Generating your personalized research plan...</small>
                    </div>
                  )}
                </Card.Body>
              </Card>
            )}

            {/* Manual Task Management */}
            <Card>
              <Card.Body>
                <h5 className="card-title mb-4">Manual Research Tasks</h5>
                
                <Form.Group className="mb-3">
                  <Form.Control
                    type="text"
                    placeholder="Add a new research task..."
                    value={newTask}
                    onChange={(e) => setNewTask(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && addTask()}
                  />
                </Form.Group>
                
                <Button variant="primary" onClick={addTask} className="mb-4">
                  <FaPlus className="me-2" />
                  Add Task
                </Button>

                {tasks.length === 0 ? (
                  <Alert variant="info">
                    No tasks yet. Add your first research task above!
                  </Alert>
                ) : (
                  <div>
                    <div className="d-flex justify-content-between align-items-center mb-3">
                      <h6 className="mb-0">Task Progress</h6>
                      <small className="text-muted">
                        {completedTasks} of {totalTasks} completed
                      </small>
                    </div>
                    <ProgressBar 
                      now={totalTasks > 0 ? (completedTasks / totalTasks) * 100 : 0} 
                      className="mb-3"
                    />
                    
                    {tasks.map(task => (
                      <Card key={task.id} className={`mb-2 ${task.completed ? 'bg-light' : ''}`}>
                        <Card.Body className="py-2">
                          <div className="d-flex align-items-center">
                            <Form.Check
                              type="checkbox"
                              checked={task.completed}
                              onChange={() => toggleTask(task.id)}
                              className="me-3"
                            />
                            <span className={`flex-grow-1 ${task.completed ? 'text-decoration-line-through text-muted' : ''}`}>
                              {task.text}
                            </span>
                            <small className="text-muted me-3">{task.date}</small>
                            <Button
                              variant="outline-danger"
                              size="sm"
                              onClick={() => deleteTask(task.id)}
                            >
                              <FaTrash />
                            </Button>
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
                    <h4 className="text-primary">{readingList?.length || 0}</h4>
                    <small className="text-muted">Papers</small>
                  </div>
                  <div className="col-6">
                    <h4 className="text-success">{completedTasks}</h4>
                    <small className="text-muted">Completed</small>
                  </div>
                </div>
              </Card.Body>
            </Card>

            {/* Current Settings */}
            <Card>
              <Card.Body>
                <h6 className="card-title">Current Settings</h6>
                <div className="small">
                  <div className="mb-2">
                    <strong>Duration:</strong> {settings.totalWeeks} weeks
                  </div>
                  <div className="mb-2">
                    <strong>Days:</strong> {settings.selectedDays.join(', ')}
                  </div>
                  <div className="mb-2">
                    <strong>Time:</strong> {settings.timeRange[0]}:00 - {settings.timeRange[1]}:00
                  </div>
                  <Button 
                    variant="outline-primary" 
                    size="sm" 
                    onClick={() => setShowSettings(true)}
                  >
                    Edit Settings
                  </Button>
                </div>
              </Card.Body>
            </Card>
          </Col>
        </Row>

        {/* Settings Modal */}
        <Modal show={showSettings} onHide={() => setShowSettings(false)}>
          <Modal.Header closeButton>
            <Modal.Title>Research Schedule Settings</Modal.Title>
          </Modal.Header>
          <Modal.Body>
            <Form>
              <Form.Group className="mb-3">
                <Form.Label>Duration (weeks)</Form.Label>
                <Form.Range
                  min="1"
                  max="12"
                  value={settings.totalWeeks}
                  onChange={(e) => setSettings({...settings, totalWeeks: parseInt(e.target.value)})}
                />
                <div className="text-center">
                  <strong>{settings.totalWeeks} weeks</strong>
                </div>
              </Form.Group>

              <Form.Group className="mb-3">
                <Form.Label>Available Days</Form.Label>
                <div className="d-flex flex-wrap gap-2">
                  {daysOfWeek.map(day => (
                    <Form.Check
                      key={day}
                      type="checkbox"
                      id={`day-${day}`}
                      label={day}
                      checked={settings.selectedDays.includes(day)}
                      onChange={(e) => {
                        if (e.target.checked) {
                          setSettings({...settings, selectedDays: [...settings.selectedDays, day]});
                        } else {
                          setSettings({...settings, selectedDays: settings.selectedDays.filter(d => d !== day)});
                        }
                      }}
                    />
                  ))}
                </div>
              </Form.Group>

              <Form.Group className="mb-3">
                <Form.Label>Available Time Range</Form.Label>
                <Form.Range
                  min="6"
                  max="22"
                  value={settings.timeRange[0]}
                  onChange={(e) => setSettings({...settings, timeRange: [parseInt(e.target.value), settings.timeRange[1]]})}
                />
                <div className="text-center">
                  <strong>{settings.timeRange[0]}:00 - {settings.timeRange[1]}:00</strong>
                </div>
              </Form.Group>
            </Form>
          </Modal.Body>
          <Modal.Footer>
            <Button variant="secondary" onClick={() => setShowSettings(false)}>
              Close
            </Button>
          </Modal.Footer>
        </Modal>

        {/* Research Plan Modal */}
        <Modal show={showPlan} onHide={() => setShowPlan(false)} size="lg">
          <Modal.Header closeButton>
            <Modal.Title>
              <FaBrain className="me-2" />
              AI-Generated Research Plan
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
              {researchPlan}
            </div>
          </Modal.Body>
          <Modal.Footer>
            <Button variant="secondary" onClick={() => setShowPlan(false)}>
              Close
            </Button>
            <Button variant="success" onClick={generateCalendarFile}>
              <FaDownload className="me-2" />
              Download Calendar
            </Button>
          </Modal.Footer>
        </Modal>
      </Container>
    </div>
  );
};

export default Planner;

import React, { useState } from 'react';
import { Container, Row, Col, Card, Form, Button, Alert } from 'react-bootstrap';
import { FaCalendarAlt, FaPlus, FaTrash, FaCheck } from 'react-icons/fa';

const Planner = () => {
  const [tasks, setTasks] = useState([]);
  const [newTask, setNewTask] = useState('');

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
            Plan and track your research activities
          </p>
        </div>

      <Row>
        <Col lg={12}>
          <Card>
            <Card.Body>
              <h5 className="card-title mb-4">Research Tasks</h5>
              
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
      </Row>
      </Container>
    </div>
  );
};

export default Planner;

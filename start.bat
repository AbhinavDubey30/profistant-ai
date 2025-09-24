@echo off
echo Starting Profistant AI...
echo.

echo Starting Backend Server...
start "Backend" cmd /k "cd backend && python app.py"

echo Waiting for backend to start...
timeout /t 3 /nobreak > nul

echo Starting Frontend...
start "Frontend" cmd /k "npm start"

echo.
echo Profistant AI is starting up!
echo Backend: http://localhost:5000
echo Frontend: http://localhost:3000
echo.
pause

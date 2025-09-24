# Profistant AI

An AI-powered research assistant for academic research that helps students and researchers find, organize, and manage research papers efficiently.

## Features

- 🔍 **Smart Research Search**: Search for academic papers using AI-powered queries
- 📚 **Reading List Management**: Save and organize papers for later reading
- 📅 **Research Planner**: Plan and track research activities and tasks
- 📊 **Dashboard**: Track research progress and insights
- 💡 **Research Ideas**: Discover and share innovative research concepts
- 📱 **Responsive Design**: Works seamlessly on desktop, tablet, and mobile
- 🎨 **Modern UI**: Beautiful, intuitive interface with smooth animations

## Tech Stack

- **Frontend**: React 18, React Router, Bootstrap, React Icons
- **Backend**: Python Flask, Scholarly API, Google Gemini AI
- **Deployment**: Vercel (Frontend + Backend)

## Getting Started

### Prerequisites

- Node.js (v14 or higher)
- Python 3.8+
- Git

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/profistant-ai.git
cd profistant-ai
```

2. Install frontend dependencies:
```bash
npm install
```

3. Install backend dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
# Create .env file in the root directory
GEMINI_API_KEY=your_gemini_api_key_here
```

5. Run the development server:
```bash
npm run dev
```

The application will be available at `http://localhost:3000`

## Deployment

This project is configured for deployment on Vercel:

1. Push your code to GitHub
2. Connect your GitHub repository to Vercel
3. Set the environment variable `GEMINI_API_KEY` in Vercel dashboard
4. Deploy!

## API Endpoints

- `GET /api/search` - Search for research papers
- `GET /api/paper/{paper_id}` - Get detailed paper information
- `POST /api/generate-ideas` - Generate research ideas using AI

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

If you have any questions or need help, please open an issue on GitHub.

---

**Made with ❤️ for the academic community**
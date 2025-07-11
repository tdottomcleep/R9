# AI Statistical Analysis App

A comprehensive AI-powered statistical analysis platform designed for medical and healthcare data analysis. Built with React, FastAPI, and MongoDB, featuring Gemini LLM integration for intelligent data insights.

## 🚀 Features

### Core Capabilities
- **📊 Medical Data Analysis**: Upload CSV files with automatic validation for healthcare datasets
- **🤖 AI-Powered Insights**: Gemini LLM integration for intelligent statistical analysis suggestions
- **🔬 Python Execution Sandbox**: Full statistical computing environment with advanced libraries
- **📋 Julius AI-Style Sectioned Execution**: Automatic code organization into logical analysis sections
- **💾 Session Management**: Persistent analysis sessions with comprehensive history tracking
- **🎨 3-Panel Interface**: Intuitive notebook-style interface with sessions, chat, and results panels

### Advanced Analytics
- **Survival Analysis**: Kaplan-Meier curves, Cox proportional hazards models
- **Epidemiological Studies**: Incidence, prevalence, mortality rates
- **Clinical Trial Analysis**: Treatment effects, safety analysis, compliance tracking
- **Biostatistics**: Descriptive statistics, hypothesis testing, regression analysis
- **Data Visualization**: Interactive plots with matplotlib, plotly, and seaborn

### Statistical Libraries Included
- **Core**: pandas, numpy, scipy, scikit-learn
- **Visualization**: matplotlib, seaborn, plotly
- **Advanced**: statsmodels, lifelines (survival analysis)
- **Specialized**: forestplot, wordcloud, bokeh, altair

## 🛠️ Installation

### Prerequisites
- **Node.js** (v16 or higher)
- **Python** (v3.8 or higher)
- **MongoDB** (v4.4 or higher)
- **Yarn** package manager

### 1. Clone the Repository
```bash
git clone <repository-url>
cd ai-statistical-analysis-app
```

### 2. Backend Setup
```bash
# Navigate to backend directory
cd backend

# Install Python dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env file with your configuration
```

### 3. Frontend Setup
```bash
# Navigate to frontend directory
cd frontend

# Install Node.js dependencies
yarn install

# Set up environment variables
cp .env.example .env
# Edit .env file with your configuration
```

### 4. Database Setup
```bash
# Start MongoDB service
sudo systemctl start mongod
# or
brew services start mongodb/brew/mongodb-community
```

### 5. Start the Application
```bash
# Terminal 1: Start Backend
cd backend
uvicorn server:app --host 0.0.0.0 --port 8001 --reload

# Terminal 2: Start Frontend
cd frontend
yarn start
```

The application will be available at:
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8001
- **API Documentation**: http://localhost:8001/docs

## 🔧 Configuration

### Environment Variables

#### Backend (.env)
```env
MONGO_URL=mongodb://localhost:27017
DB_NAME=statistical_analysis_db
GEMINI_API_KEY=your_gemini_api_key_here
```

#### Frontend (.env)
```env
REACT_APP_BACKEND_URL=http://localhost:8001
```

### Gemini API Key Setup
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Add it to your backend `.env` file as `GEMINI_API_KEY`

## 📖 Usage Guide

### 1. Upload CSV Data
- Click "Upload CSV" in the left panel
- Select a CSV file containing your data
- The system will validate and create a new analysis session

### 2. Interactive Analysis
- Use the chat interface to ask questions about your data
- Request specific statistical analyses
- Get AI-powered suggestions for appropriate tests

### 3. Code Execution
- Execute Python code directly in the sandbox
- Access your uploaded data as `df` variable
- View results, plots, and outputs in the right panel

### 4. Sectioned Analysis
- Use Julius AI-style sectioned execution for organized analysis
- Automatic classification of code sections (summary, statistical tests, visualizations)
- Structured results with metadata and context

### 5. Analysis History
- View all previous analyses and results
- Access structured analysis results
- Track your analytical workflow

## 🔌 API Documentation

### Core Endpoints

#### Sessions
- `GET /api/sessions` - List all sessions
- `POST /api/sessions` - Create new session (with CSV upload)
- `GET /api/sessions/{id}` - Get session details
- `GET /api/sessions/{id}/messages` - Get session messages

#### Analysis
- `POST /api/sessions/{id}/chat` - Chat with LLM
- `POST /api/sessions/{id}/execute` - Execute Python code
- `POST /api/sessions/{id}/execute-sectioned` - Execute with sectioned analysis
- `POST /api/sessions/{id}/suggest-analysis` - Get analysis suggestions

#### Structured Analysis
- `GET /api/sessions/{id}/structured-analyses` - List structured analyses
- `GET /api/sessions/{id}/structured-analyses/{analysis_id}` - Get specific analysis

#### History
- `GET /api/sessions/{id}/analysis-history` - Get analysis history

### Request/Response Examples

#### Upload CSV and Create Session
```bash
curl -X POST "http://localhost:8001/api/sessions" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_data.csv"
```

#### Chat with LLM
```bash
curl -X POST "http://localhost:8001/api/sessions/{session_id}/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What statistical tests would you recommend for this data?",
    "gemini_api_key": "your_gemini_api_key"
  }'
```

#### Execute Python Code
```bash
curl -X POST "http://localhost:8001/api/sessions/{session_id}/execute" \
  -H "Content-Type: application/json" \
  -d '{
    "code": "df.describe()"
  }'
```

## 🏗️ Architecture

### Project Structure
```
ai-statistical-analysis-app/
├── backend/                    # FastAPI Backend
│   ├── server.py              # Main application
│   ├── requirements.txt       # Python dependencies
│   ├── .env                   # Environment variables
│   └── .env.example           # Environment template
├── frontend/                   # React Frontend
│   ├── src/
│   │   ├── App.js             # Main React component
│   │   ├── App.css            # Styling
│   │   └── index.js           # Entry point
│   ├── public/                # Static assets
│   ├── package.json           # Node.js dependencies
│   ├── .env                   # Environment variables
│   └── .env.example           # Environment template
├── examples/                   # Sample datasets
│   ├── sample_medical_data.csv
│   ├── clinical_trial_data.csv
│   ├── survival_data.csv
│   └── README.md              # Dataset documentation
├── docs/                       # Documentation
│   ├── README.md              # Main documentation
│   ├── INSTALLATION.md        # Setup guide
│   ├── USAGE_GUIDE.md         # User guide
│   └── API_REFERENCE.md       # API documentation
├── project-compass.json       # Project blueprint
└── setup.sh                   # Quick setup script
```

### Key Components

#### Backend Classes
- `AnalysisClassifier` - Intelligent code section classification
- `JuliusStyleExecutor` - Sectioned execution management
- `AnalysisSection` - Individual analysis section model
- `StructuredAnalysisResult` - Complete analysis results

#### Frontend Components
- **Left Panel**: Session management and file upload
- **Center Panel**: Chat interface and message history
- **Right Panel**: Execution results and analysis history

## 🔒 Security Notes

- API keys are stored securely in environment variables
- File uploads are validated for CSV format
- Code execution runs in a controlled sandbox environment
- All API requests are validated and sanitized

## 🐛 Troubleshooting

### Common Issues

#### Backend Won't Start
```bash
# Check Python dependencies
pip install -r requirements.txt

# Verify MongoDB is running
sudo systemctl status mongod

# Check environment variables
cat backend/.env
```

#### Frontend Connection Issues
```bash
# Verify backend URL in frontend/.env
echo $REACT_APP_BACKEND_URL

# Check CORS settings
curl -H "Origin: http://localhost:3000" \
  -H "Access-Control-Request-Method: GET" \
  -H "Access-Control-Request-Headers: X-Requested-With" \
  -X OPTIONS http://localhost:8001/api/sessions
```

#### Gemini API Issues
- Verify API key is valid and has proper permissions
- Check API quota and rate limits
- Ensure network connectivity to Google AI services

## 📊 Example Use Cases

### Medical Research
- Analyze clinical trial data
- Perform survival analysis on patient outcomes
- Calculate epidemiological metrics
- Statistical testing for treatment efficacy

### Healthcare Analytics
- Patient demographic analysis
- Treatment outcome comparisons
- Risk factor identification
- Diagnostic test performance evaluation

### General Statistics
- Descriptive statistics and data exploration
- Hypothesis testing and confidence intervals
- Regression analysis and modeling
- Data visualization and reporting

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues, questions, or feature requests:
1. Check the troubleshooting section
2. Search existing issues on GitHub
3. Create a new issue with detailed information
4. Include error messages and system information

---

**Built with ❤️ for the healthcare and research community**
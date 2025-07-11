# Installation Guide

## Quick Start

### Prerequisites
- **Node.js** v16+ ([Download](https://nodejs.org/))
- **Python** 3.8+ ([Download](https://python.org/))
- **MongoDB** 4.4+ ([Download](https://www.mongodb.com/try/download/community))
- **Yarn** package manager ([Install](https://yarnpkg.com/getting-started/install))

### One-Command Setup (Recommended)

```bash
# Clone and setup everything
git clone <repository-url>
cd ai-statistical-analysis-app
./setup.sh  # Coming soon
```

### Manual Setup

#### 1. Clone Repository
```bash
git clone <repository-url>
cd ai-statistical-analysis-app
```

#### 2. Backend Setup
```bash
cd backend

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup environment
cp .env.example .env
# Edit .env with your configuration
```

#### 3. Frontend Setup
```bash
cd frontend

# Install dependencies
yarn install

# Setup environment
cp .env.example .env
# Edit .env with your configuration
```

#### 4. Database Setup
```bash
# Start MongoDB
sudo systemctl start mongod
# or on macOS:
brew services start mongodb/brew/mongodb-community

# Verify MongoDB is running
mongo --eval "db.stats()"
```

#### 5. Start Services
```bash
# Terminal 1: Backend
cd backend
uvicorn server:app --host 0.0.0.0 --port 8001 --reload

# Terminal 2: Frontend
cd frontend
yarn start
```

## Environment Configuration

### Backend Environment (.env)
```env
# Database
MONGO_URL=mongodb://localhost:27017
DB_NAME=statistical_analysis_db

# AI Integration
GEMINI_API_KEY=your_gemini_api_key_here

# Optional
STRIPE_API_KEY=sk_test_your_stripe_key
```

### Frontend Environment (.env)
```env
# Backend API
REACT_APP_BACKEND_URL=http://localhost:8001

# Development
WDS_SOCKET_PORT=3000
```

## API Key Setup

### Gemini API Key (Required)
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the generated key
5. Add to backend `.env` file:
   ```env
   GEMINI_API_KEY=your_actual_key_here
   ```

## Platform-Specific Instructions

### Windows
```bash
# Install Python dependencies
pip install -r requirements.txt

# Start MongoDB (if installed via MSI)
net start MongoDB

# Alternative: MongoDB service
services.msc -> MongoDB -> Start
```

### macOS
```bash
# Install MongoDB via Homebrew
brew tap mongodb/brew
brew install mongodb-community

# Start MongoDB
brew services start mongodb/brew/mongodb-community

# Install Python dependencies
pip3 install -r requirements.txt
```

### Linux (Ubuntu/Debian)
```bash
# Install MongoDB
sudo apt-get install mongodb

# Start MongoDB
sudo systemctl start mongod
sudo systemctl enable mongod

# Install Python dependencies
pip3 install -r requirements.txt
```

## Docker Setup (Alternative)

### Using Docker Compose
```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Docker Compose File
```yaml
version: '3.8'
services:
  mongodb:
    image: mongo:4.4
    ports:
      - "27017:27017"
    volumes:
      - mongodb_data:/data/db
  
  backend:
    build: ./backend
    ports:
      - "8001:8001"
    environment:
      - MONGO_URL=mongodb://mongodb:27017
    depends_on:
      - mongodb
  
  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    environment:
      - REACT_APP_BACKEND_URL=http://localhost:8001
    depends_on:
      - backend

volumes:
  mongodb_data:
```

## Verification

### Check Installation
```bash
# Backend health check
curl http://localhost:8001/api/health

# Frontend accessibility
curl http://localhost:3000

# Database connection
mongo --eval "db.runCommand('ping')"
```

### Test Upload
1. Open http://localhost:3000
2. Click "Upload CSV"
3. Select a CSV file
4. Verify session creation
5. Test chat functionality

## Troubleshooting

### Common Issues

#### MongoDB Connection Error
```bash
# Check MongoDB status
sudo systemctl status mongod

# Check port availability
netstat -tulpn | grep :27017

# Restart MongoDB
sudo systemctl restart mongod
```

#### Python Dependencies Issues
```bash
# Update pip
pip install --upgrade pip

# Clear cache and reinstall
pip cache purge
pip install -r requirements.txt --force-reinstall
```

#### Node.js Dependencies Issues
```bash
# Clear cache
yarn cache clean

# Delete node_modules and reinstall
rm -rf node_modules
yarn install
```

#### API Key Issues
- Verify API key format (starts with "AIza")
- Check API key permissions
- Ensure billing is enabled for Gemini API
- Test key with curl:
  ```bash
  curl -X POST \
    -H "Content-Type: application/json" \
    -d '{"contents":[{"parts":[{"text":"Hello"}]}]}' \
    "https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent?key=YOUR_API_KEY"
  ```

### Performance Optimization

#### Backend
```bash
# Use production WSGI server
pip install gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker server:app
```

#### Frontend
```bash
# Build production version
yarn build

# Serve with nginx or Apache
```

## Next Steps

1. **Test the application** with sample data
2. **Configure your AI API keys** for full functionality
3. **Explore the API documentation** at http://localhost:8001/docs
4. **Review the user guide** for advanced features
5. **Check the troubleshooting section** if you encounter issues

## Support

- **Documentation**: Check README.md for detailed usage
- **API Reference**: http://localhost:8001/docs
- **Issues**: Create GitHub issues for bugs
- **Community**: Join our discussion forum
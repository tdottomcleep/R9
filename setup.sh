#!/bin/bash

# AI Statistical Analysis App - Quick Setup Script
# This script will set up the entire application with one command

set -e  # Exit on any error

echo "🚀 AI Statistical Analysis App Setup"
echo "===================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Check if running on supported OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    OS="windows"
else
    print_error "Unsupported operating system: $OSTYPE"
    exit 1
fi

print_status "Detected OS: $OS"

# Check prerequisites
print_step "Checking prerequisites..."

# Check Python
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

python_version=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1-2)
print_status "Python version: $python_version"

# Check Node.js
if ! command -v node &> /dev/null; then
    print_error "Node.js is not installed. Please install Node.js 16 or higher."
    exit 1
fi

node_version=$(node --version)
print_status "Node.js version: $node_version"

# Check Yarn
if ! command -v yarn &> /dev/null; then
    print_warning "Yarn is not installed. Installing yarn..."
    npm install -g yarn
fi

yarn_version=$(yarn --version)
print_status "Yarn version: $yarn_version"

# Check MongoDB
if ! command -v mongod &> /dev/null; then
    print_warning "MongoDB is not installed or not in PATH."
    print_warning "Please install MongoDB and ensure it's running."
    print_warning "Installation guides:"
    print_warning "- Linux: https://docs.mongodb.com/manual/installation/"
    print_warning "- macOS: brew install mongodb/brew/mongodb-community"
    print_warning "- Windows: https://docs.mongodb.com/manual/tutorial/install-mongodb-on-windows/"
    echo ""
    read -p "Do you want to continue setup without MongoDB check? (y/N): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    print_status "MongoDB found"
fi

# Start setup
print_step "Starting application setup..."

# Create project directory structure
print_step "Creating project structure..."
mkdir -p logs
mkdir -p data
mkdir -p temp

# Backend setup
print_step "Setting up backend..."
cd backend

# Create virtual environment
print_status "Creating Python virtual environment..."
python3 -m venv venv

# Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate

# Install Python dependencies
print_status "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Setup backend environment
if [ ! -f .env ]; then
    print_status "Creating backend .env file..."
    cp .env.example .env
    print_warning "Please edit backend/.env with your configuration"
    print_warning "Especially set your GEMINI_API_KEY"
fi

cd ..

# Frontend setup
print_step "Setting up frontend..."
cd frontend

# Install Node.js dependencies
print_status "Installing Node.js dependencies..."
yarn install

# Setup frontend environment
if [ ! -f .env ]; then
    print_status "Creating frontend .env file..."
    cp .env.example .env
fi

cd ..

# Create startup scripts
print_step "Creating startup scripts..."

# Backend startup script
cat > start_backend.sh << 'EOF'
#!/bin/bash
cd backend
source venv/bin/activate
uvicorn server:app --host 0.0.0.0 --port 8001 --reload
EOF

# Frontend startup script
cat > start_frontend.sh << 'EOF'
#!/bin/bash
cd frontend
yarn start
EOF

# Combined startup script
cat > start_app.sh << 'EOF'
#!/bin/bash
echo "Starting AI Statistical Analysis App..."
echo "Backend will be available at: http://localhost:8001"
echo "Frontend will be available at: http://localhost:3000"
echo "Press Ctrl+C to stop all services"
echo ""

# Start backend in background
echo "Starting backend..."
cd backend
source venv/bin/activate
uvicorn server:app --host 0.0.0.0 --port 8001 --reload &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 3

# Start frontend in background
echo "Starting frontend..."
cd ../frontend
yarn start &
FRONTEND_PID=$!

# Function to cleanup processes on exit
cleanup() {
    echo ""
    echo "Shutting down services..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    exit 0
}

# Register cleanup function
trap cleanup SIGINT SIGTERM

# Wait for processes
wait $BACKEND_PID $FRONTEND_PID
EOF

# Make scripts executable
chmod +x start_backend.sh start_frontend.sh start_app.sh

# MongoDB setup
print_step "Setting up MongoDB..."

# Start MongoDB if not running
if command -v mongod &> /dev/null; then
    if ! pgrep mongod > /dev/null; then
        print_status "Starting MongoDB..."
        if [[ "$OS" == "linux" ]]; then
            sudo systemctl start mongod
        elif [[ "$OS" == "macos" ]]; then
            brew services start mongodb/brew/mongodb-community
        fi
    else
        print_status "MongoDB is already running"
    fi
fi

# Test MongoDB connection
print_status "Testing MongoDB connection..."
if command -v mongo &> /dev/null; then
    if mongo --eval "db.runCommand('ping')" --quiet; then
        print_status "MongoDB connection successful"
    else
        print_warning "MongoDB connection failed"
    fi
fi

# Final setup steps
print_step "Final setup steps..."

# Create README with quick start
cat > QUICK_START.md << 'EOF'
# Quick Start Guide

## Start the Application
```bash
./start_app.sh
```

## Individual Services
```bash
# Backend only
./start_backend.sh

# Frontend only
./start_frontend.sh
```

## Access Points
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8001
- **API Documentation**: http://localhost:8001/docs

## First Steps
1. Open http://localhost:3000 in your browser
2. Click "Upload CSV" and select a file from the `examples/` directory
3. Start chatting with the AI assistant
4. Execute Python code in the sandbox

## Need Help?
- Check the `INSTALLATION.md` for detailed setup
- Review `USAGE_GUIDE.md` for features
- See `API_REFERENCE.md` for API details
EOF

# Setup complete
echo ""
echo "✅ Setup Complete!"
echo "=================="
echo ""
print_status "Application setup completed successfully!"
echo ""
echo "📂 Next steps:"
echo "1. Edit backend/.env with your Gemini API key"
echo "2. Run: ./start_app.sh"
echo "3. Open http://localhost:3000 in your browser"
echo ""
echo "📖 Documentation:"
echo "- Quick Start: ./QUICK_START.md"
echo "- Full Guide: ./USAGE_GUIDE.md"
echo "- API Reference: ./API_REFERENCE.md"
echo ""
echo "🔧 Individual services:"
echo "- Backend only: ./start_backend.sh"
echo "- Frontend only: ./start_frontend.sh"
echo ""
echo "🎯 Test with sample data:"
echo "- Upload files from the examples/ directory"
echo "- Try medical data analysis workflows"
echo ""
print_status "Happy analyzing! 🔬📊"
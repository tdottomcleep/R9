from fastapi import FastAPI, APIRouter, HTTPException, File, UploadFile, Form
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import json
import asyncio
import subprocess
import sys
import tempfile
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# Emergent LLM integration
from emergentintegrations.llm.chat import LlmChat, UserMessage, FileContentWithMimeType

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Models
class ChatSession(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    file_name: str
    file_data: Optional[str] = None  # Base64 encoded CSV data
    csv_preview: Optional[Dict] = None

class ChatMessage(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    analysis_result: Optional[Dict] = None

class APIKeyConfig(BaseModel):
    gemini_api_key: str

class AnalysisRequest(BaseModel):
    session_id: str
    analysis_type: str
    columns: List[str]
    gemini_api_key: str

class PythonExecutionRequest(BaseModel):
    session_id: str
    code: str
    gemini_api_key: str

# API Routes
@api_router.get("/")
async def root():
    return {"message": "AI Data Scientist API"}

@api_router.post("/sessions")
async def create_session(file: UploadFile = File(...)):
    """Create a new chat session with CSV file upload"""
    try:
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are supported")
        
        # Read and validate CSV
        content = await file.read()
        try:
            df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid CSV file: {str(e)}")
        
        # Create preview data
        preview = {
            "columns": df.columns.tolist(),
            "shape": df.shape,
            "head": df.head().to_dict('records'),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "null_counts": df.isnull().sum().to_dict(),
            "describe": df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {}
        }
        
        # Create session
        session = ChatSession(
            title=file.filename,
            file_name=file.filename,
            file_data=base64.b64encode(content).decode('utf-8'),
            csv_preview=preview
        )
        
        # Save to database
        await db.chat_sessions.insert_one(session.dict())
        
        return session
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/sessions")
async def get_sessions():
    """Get all chat sessions"""
    sessions = await db.chat_sessions.find().sort("created_at", -1).to_list(100)
    return [ChatSession(**session) for session in sessions]

@api_router.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Get a specific session"""
    session = await db.chat_sessions.find_one({"id": session_id})
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return ChatSession(**session)

@api_router.get("/sessions/{session_id}/messages")
async def get_messages(session_id: str):
    """Get messages for a session"""
    messages = await db.chat_messages.find({"session_id": session_id}).sort("timestamp", 1).to_list(1000)
    return [ChatMessage(**message) for message in messages]

@api_router.post("/sessions/{session_id}/chat")
async def chat_with_llm(session_id: str, message: str = Form(...), gemini_api_key: str = Form(...)):
    """Chat with LLM about the data"""
    try:
        # Get session
        session = await db.chat_sessions.find_one({"id": session_id})
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Save user message
        user_message = ChatMessage(
            session_id=session_id,
            role="user",
            content=message
        )
        await db.chat_messages.insert_one(user_message.dict())
        
        # Prepare context for LLM
        csv_preview = session.get('csv_preview', {})
        context = f"""
        You are an AI Data Scientist assistant. The user has uploaded a CSV file: {session['file_name']}
        
        Dataset Info:
        - Shape: {csv_preview.get('shape', 'Unknown')}
        - Columns: {csv_preview.get('columns', [])}
        - Data Types: {csv_preview.get('dtypes', {})}
        - Null Values: {csv_preview.get('null_counts', {})}
        
        Sample Data:
        {csv_preview.get('head', [])}
        
        Statistical Summary:
        {csv_preview.get('describe', {})}
        
        You should:
        1. Help analyze the medical/statistical data
        2. Suggest appropriate statistical tests
        3. Provide insights about the data
        4. Offer to run specific analyses
        5. Generate Python code for statistical analysis when requested
        
        When suggesting analysis, be specific about what columns to use and what tests are appropriate.
        """
        
        # Chat with Gemini
        chat = LlmChat(
            api_key=gemini_api_key,
            session_id=session_id,
            system_message=context
        ).with_model("gemini", "gemini-2.5-pro-preview-05-06")
        
        response = await chat.send_message(UserMessage(text=message))
        
        # Save assistant response
        assistant_message = ChatMessage(
            session_id=session_id,
            role="assistant",
            content=response
        )
        await db.chat_messages.insert_one(assistant_message.dict())
        
        return {"response": response}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/sessions/{session_id}/execute")
async def execute_python_code(session_id: str, request: PythonExecutionRequest):
    """Execute Python code for statistical analysis"""
    try:
        # Get session data
        session = await db.chat_sessions.find_one({"id": session_id})
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Decode CSV data
        csv_data = base64.b64decode(session['file_data']).decode('utf-8')
        df = pd.read_csv(io.StringIO(csv_data))
        
        # Prepare execution environment
        execution_globals = {
            'df': df,
            'pd': pd,
            'np': np,
            'plt': plt,
            'sns': sns,
            'stats': stats,
            'LinearRegression': LinearRegression,
            'r2_score': r2_score,
            'io': io,
            'base64': base64
        }
        
        # Capture output
        output_buffer = io.StringIO()
        error_buffer = io.StringIO()
        
        try:
            # Redirect stdout
            old_stdout = sys.stdout
            sys.stdout = output_buffer
            
            # Execute code
            exec(request.code, execution_globals)
            
            # Get output
            output = output_buffer.getvalue()
            
            # Handle matplotlib plots
            plots = []
            if plt.get_fignums():
                for fig_num in plt.get_fignums():
                    fig = plt.figure(fig_num)
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
                    buf.seek(0)
                    plot_data = base64.b64encode(buf.read()).decode('utf-8')
                    plots.append(plot_data)
                    buf.close()
                plt.close('all')
            
            result = {
                "success": True,
                "output": output,
                "plots": plots,
                "error": None
            }
            
        except Exception as e:
            result = {
                "success": False,
                "output": output_buffer.getvalue(),
                "plots": [],
                "error": str(e)
            }
        
        finally:
            sys.stdout = old_stdout
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/sessions/{session_id}/suggest-analysis")
async def suggest_analysis(session_id: str, gemini_api_key: str = Form(...)):
    """Get analysis suggestions from LLM"""
    try:
        # Get session
        session = await db.chat_sessions.find_one({"id": session_id})
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        csv_preview = session.get('csv_preview', {})
        
        # Ask LLM for analysis suggestions
        context = f"""
        You are an AI Data Scientist. Analyze this dataset and suggest appropriate statistical analyses.
        
        Dataset: {session['file_name']}
        Shape: {csv_preview.get('shape', 'Unknown')}
        Columns: {csv_preview.get('columns', [])}
        Data Types: {csv_preview.get('dtypes', {})}
        
        Sample Data:
        {csv_preview.get('head', [])}
        
        Please suggest 3-5 specific statistical analyses that would be appropriate for this medical/research data.
        For each suggestion, provide:
        1. Analysis name
        2. Brief description
        3. Which columns to use
        4. What insights it would provide
        
        Format your response as a JSON list of suggestions.
        """
        
        chat = LlmChat(
            api_key=gemini_api_key,
            session_id=f"{session_id}_suggestions",
            system_message="You are a statistical analysis expert. Provide suggestions in JSON format."
        ).with_model("gemini", "gemini-2.5-pro-preview-05-06")
        
        response = await chat.send_message(UserMessage(text=context))
        
        return {"suggestions": response}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
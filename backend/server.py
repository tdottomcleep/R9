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
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
import plotly.io as pio
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
import statsmodels.api as sm
from statsmodels.stats.contingency_tables import mcnemar
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
import warnings
warnings.filterwarnings('ignore')

# New models for Julius AI-style sectioned analysis
class AnalysisSection(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    section_type: str  # 'summary', 'descriptive', 'statistical_test', 'visualization', 'model'
    code: str
    output: str
    success: bool
    error: Optional[str] = None
    metadata: Dict = Field(default_factory=dict)
    tables: List[Dict] = Field(default_factory=list)
    charts: List[Dict] = Field(default_factory=list)
    order: int = 0

class StructuredAnalysisResult(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    title: str
    sections: List[AnalysisSection]
    total_sections: int
    execution_time: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    overall_success: bool

# Analysis Classification System for Julius AI-style sections
class AnalysisClassifier:
    
    @staticmethod
    def classify_code_section(code: str) -> tuple[str, str]:
        """Classify a code section and return (section_type, title)"""
        code_lower = code.lower()
        
        # Summary/Overview patterns
        if any(pattern in code_lower for pattern in [
            'summary', 'overview', 'describe', 'info()', 'head()', 'shape',
            'clinical outcomes', 'total patients', 'key findings'
        ]):
            return 'summary', 'Clinical Overview Summary'
        
        # Descriptive statistics patterns
        elif any(pattern in code_lower for pattern in [
            'mean()', 'median()', 'std()', 'var()', 'quantile()',
            'describe()', 'value_counts()', 'groupby', 'agg('
        ]):
            return 'descriptive', 'Descriptive Statistics'
        
        # Statistical test patterns
        elif any(pattern in code_lower for pattern in [
            'ttest', 'chi2', 'anova', 'mannwhitney', 'wilcoxon',
            'pearson', 'spearman', 'correlation', 'regression',
            'logistic', 'pvalue', 'significance'
        ]):
            return 'statistical_test', 'Statistical Testing'
        
        # Visualization patterns
        elif any(pattern in code_lower for pattern in [
            'plot', 'hist', 'scatter', 'bar', 'pie', 'box',
            'seaborn', 'matplotlib', 'plotly', 'visualization'
        ]):
            return 'visualization', 'Data Visualization'
        
        # Model building patterns
        elif any(pattern in code_lower for pattern in [
            'model', 'fit()', 'predict', 'score', 'cross_val',
            'randomforest', 'svm', 'neural', 'machine learning'
        ]):
            return 'model', 'Model Analysis'
        
        # Data processing patterns
        elif any(pattern in code_lower for pattern in [
            'fillna', 'dropna', 'merge', 'join', 'transform',
            'encode', 'scale', 'normalize', 'preprocess'
        ]):
            return 'preprocessing', 'Data Preprocessing'
        
        # Default classification
        else:
            return 'analysis', 'Data Analysis'
    
    @staticmethod
    def extract_tables_from_output(output: str, execution_globals: dict) -> List[Dict]:
        """Extract table data from execution output and globals"""
        tables = []
        
        # Look for pandas DataFrames in output
        lines = output.split('\n')
        current_table = []
        in_table = False
        
        for line in lines:
            # Detect DataFrame-like output
            if ('|' in line and len(line.split('|')) > 2) or (line.strip() and line.strip()[0].isdigit()):
                if not in_table:
                    in_table = True
                    current_table = []
                current_table.append(line)
            else:
                if in_table and current_table:
                    # Process completed table
                    table_text = '\n'.join(current_table)
                    if len(current_table) > 1:  # Valid table
                        tables.append({
                            'type': 'dataframe',
                            'title': 'Data Table',
                            'content': table_text,
                            'rows': len(current_table) - 1,  # Exclude header
                            'clickable': True
                        })
                    in_table = False
                    current_table = []
        
        # Check for DataFrames in execution globals
        for var_name, var_value in execution_globals.items():
            if isinstance(var_value, pd.DataFrame) and not var_name.startswith('_'):
                tables.append({
                    'type': 'dataframe',
                    'title': f'Table: {var_name}',
                    'content': var_value.to_string(),
                    'data': var_value.head(10).to_dict('records'),  # First 10 rows
                    'shape': var_value.shape,
                    'clickable': True
                })
        
        return tables
    
    @staticmethod
    def determine_chart_type(code: str, output: str) -> str:
        """Determine appropriate chart type based on code and output"""
        code_lower = code.lower()
        
        if any(pattern in code_lower for pattern in ['pie', 'value_counts']):
            return 'pie'
        elif any(pattern in code_lower for pattern in ['hist', 'distribution']):
            return 'histogram'
        elif any(pattern in code_lower for pattern in ['bar', 'count']):
            return 'bar'
        elif any(pattern in code_lower for pattern in ['scatter', 'correlation']):
            return 'scatter'
        elif any(pattern in code_lower for pattern in ['box', 'quartile']):
            return 'box'
        elif any(pattern in code_lower for pattern in ['line', 'time', 'trend']):
            return 'line'
        else:
            return 'bar'  # Default

# Enhanced Python execution engine
class JuliusStyleExecutor:
    
    def __init__(self, session_data: dict):
        self.session_data = session_data
        self.df = None
        self.execution_globals = {}
        self._setup_execution_environment()
    
    def _setup_execution_environment(self):
        """Setup the execution environment with data and libraries"""
        # Decode CSV data
        if self.session_data.get('file_data'):
            csv_data = base64.b64decode(self.session_data['file_data']).decode('utf-8')
            self.df = pd.read_csv(io.StringIO(csv_data))
        
        # Setup execution globals
        self.execution_globals = {
            'df': self.df,
            'pd': pd,
            'np': np,
            'plt': plt,
            'sns': sns,
            'stats': stats,
            'LinearRegression': LinearRegression,
            'r2_score': r2_score,
            'io': io,
            'base64': base64,
            'go': go,
            'px': px,
            'ff': ff,
            'pio': pio,
            'sm': sm,
            'mcnemar': mcnemar,
            'KaplanMeierFitter': KaplanMeierFitter,
            'CoxPHFitter': CoxPHFitter,
            'logrank_test': logrank_test,
            'datetime': datetime,
            'json': json
        }
    
    def execute_sectioned_analysis(self, code: str, analysis_title: str = "Statistical Analysis") -> StructuredAnalysisResult:
        """Execute code and return structured analysis sections"""
        start_time = datetime.utcnow()
        
        # Split code into logical sections
        sections = self._split_code_into_sections(code)
        
        analysis_sections = []
        overall_success = True
        
        for i, (section_code, section_title) in enumerate(sections):
            section_result = self._execute_single_section(section_code, section_title, i)
            analysis_sections.append(section_result)
            
            if not section_result.success:
                overall_success = False
        
        end_time = datetime.utcnow()
        execution_time = (end_time - start_time).total_seconds()
        
        return StructuredAnalysisResult(
            session_id=self.session_data['id'],
            title=analysis_title,
            sections=analysis_sections,
            total_sections=len(analysis_sections),
            execution_time=execution_time,
            overall_success=overall_success
        )
    
    def _split_code_into_sections(self, code: str) -> List[tuple[str, str]]:
        """Split code into logical sections with titles"""
        sections = []
        
        # Split by comments or logical breaks
        lines = code.split('\n')
        current_section = []
        current_title = "Analysis"
        
        for line in lines:
            stripped = line.strip()
            
            # Check for section headers (comments that look like titles)
            if stripped.startswith('#') and len(stripped) > 3:
                if current_section:
                    # Process previous section
                    section_code = '\n'.join(current_section)
                    if section_code.strip():
                        section_type, auto_title = AnalysisClassifier.classify_code_section(section_code)
                        sections.append((section_code, current_title if current_title != "Analysis" else auto_title))
                    current_section = []
                
                # Extract title from comment
                current_title = stripped[1:].strip()
                if current_title.startswith('='):
                    current_title = current_title.strip('=').strip()
            else:
                current_section.append(line)
        
        # Add final section
        if current_section:
            section_code = '\n'.join(current_section)
            if section_code.strip():
                section_type, auto_title = AnalysisClassifier.classify_code_section(section_code)
                sections.append((section_code, current_title if current_title != "Analysis" else auto_title))
        
        # If no sections found, treat entire code as one section
        if not sections:
            section_type, auto_title = AnalysisClassifier.classify_code_section(code)
            sections.append((code, auto_title))
        
        return sections
    
    def _execute_single_section(self, code: str, title: str, order: int) -> AnalysisSection:
        """Execute a single code section"""
        section_type, _ = AnalysisClassifier.classify_code_section(code)
        
        # Capture output
        output_buffer = io.StringIO()
        old_stdout = sys.stdout
        
        try:
            sys.stdout = output_buffer
            
            # Execute code
            exec(code, self.execution_globals)
            
            # Get output
            output = output_buffer.getvalue()
            
            # Extract tables
            tables = AnalysisClassifier.extract_tables_from_output(output, self.execution_globals)
            
            # Handle plots
            charts = self._extract_charts(code)
            
            # Create metadata
            metadata = {
                'lines_of_code': len(code.split('\n')),
                'has_output': bool(output.strip()),
                'chart_type': AnalysisClassifier.determine_chart_type(code, output),
                'variables_used': self._extract_variables_used(code)
            }
            
            return AnalysisSection(
                title=title,
                section_type=section_type,
                code=code,
                output=output,
                success=True,
                error=None,
                metadata=metadata,
                tables=tables,
                charts=charts,
                order=order
            )
            
        except Exception as e:
            return AnalysisSection(
                title=title,
                section_type=section_type,
                code=code,
                output=output_buffer.getvalue(),
                success=False,
                error=str(e),
                metadata={'error_type': type(e).__name__},
                tables=[],
                charts=[],
                order=order
            )
        finally:
            sys.stdout = old_stdout
    
    def _extract_charts(self, code: str) -> List[Dict]:
        """Extract chart data from matplotlib and plotly"""
        charts = []
        
        # Handle matplotlib plots
        if plt.get_fignums():
            for fig_num in plt.get_fignums():
                fig = plt.figure(fig_num)
                buf = io.BytesIO()
                fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
                buf.seek(0)
                plot_data = base64.b64encode(buf.read()).decode('utf-8')
                
                chart_type = AnalysisClassifier.determine_chart_type(code, "")
                charts.append({
                    'type': 'matplotlib',
                    'chart_type': chart_type,
                    'data': plot_data,
                    'title': f'{chart_type.title()} Chart',
                    'clickable': True
                })
                buf.close()
            plt.close('all')
        
        # Handle Plotly plots
        for var_name, var_value in self.execution_globals.items():
            if hasattr(var_value, '_module') and 'plotly' in str(var_value._module):
                try:
                    html_str = var_value.to_html(include_plotlyjs='cdn')
                    chart_type = AnalysisClassifier.determine_chart_type(code, "")
                    charts.append({
                        'type': 'plotly',
                        'chart_type': chart_type,
                        'html': html_str,
                        'title': f'{chart_type.title()} Chart',
                        'clickable': True
                    })
                except:
                    pass
        
        return charts
    
    def _extract_variables_used(self, code: str) -> List[str]:
        """Extract variable names used in code"""
        import re
        
        # Simple regex to find DataFrame column references
        df_columns = re.findall(r"df\[['\"](.*?)['\"]\]", code)
        df_attrs = re.findall(r"df\.(\w+)", code)
        
        variables = list(set(df_columns + df_attrs))
        return [var for var in variables if var not in ['head', 'tail', 'shape', 'info', 'describe']]

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

class AnalysisResult(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: Optional[str] = None
    analysis_type: str
    variables: List[str]
    test_statistic: Optional[float] = None
    p_value: Optional[float] = None
    effect_size: Optional[float] = None
    confidence_interval: Optional[List[float]] = None
    interpretation: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    raw_results: Optional[Dict] = None

# New models for Julius AI-style sectioned analysis
class AnalysisSection(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    section_type: str  # 'summary', 'descriptive', 'statistical_test', 'visualization', 'model'
    code: str
    output: str
    success: bool
    error: Optional[str] = None
    metadata: Dict = Field(default_factory=dict)
    tables: List[Dict] = Field(default_factory=list)
    charts: List[Dict] = Field(default_factory=list)
    order: int = 0

class StructuredAnalysisResult(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    title: str
    sections: List[AnalysisSection]
    total_sections: int
    execution_time: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    overall_success: bool

class EnhancedPythonExecutionRequest(BaseModel):
    session_id: str
    code: str
    gemini_api_key: str
    analysis_title: Optional[str] = "Statistical Analysis"
    auto_section: bool = True  # Whether to auto-detect analysis sections

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
        
        # Prepare enhanced context for LLM
        csv_preview = session.get('csv_preview', {})
        
        # Enhanced data analysis context
        columns = csv_preview.get('columns', [])
        dtypes = csv_preview.get('dtypes', {})
        null_counts = csv_preview.get('null_counts', {})
        describe_stats = csv_preview.get('describe', {})
        
        # Analyze data types and potential study design
        numeric_cols = [col for col, dtype in dtypes.items() if 'int' in str(dtype) or 'float' in str(dtype)]
        categorical_cols = [col for col, dtype in dtypes.items() if 'object' in str(dtype)]
        
        # Identify potential study variables
        potential_outcomes = []
        potential_exposures = []
        potential_time_vars = []
        
        for col in columns:
            col_lower = col.lower()
            # Identify potential outcome variables
            if any(term in col_lower for term in ['outcome', 'death', 'survival', 'event', 'response', 'improvement', 'cure']):
                potential_outcomes.append(col)
            # Identify potential exposure/treatment variables
            elif any(term in col_lower for term in ['treatment', 'group', 'arm', 'intervention', 'therapy', 'drug', 'placebo', 'vaccine']):
                potential_exposures.append(col)
            # Identify potential time variables
            elif any(term in col_lower for term in ['time', 'day', 'week', 'month', 'year', 'duration', 'follow']):
                potential_time_vars.append(col)
        
        context = f"""
        You are an Expert AI Data Scientist and Biostatistician. You have been provided with a medical/research dataset: {session['file_name']}
        
        DATASET OVERVIEW:
        - Shape: {csv_preview.get('shape', 'Unknown')} (rows × columns)
        - Total Variables: {len(columns)}
        - Numeric Variables: {len(numeric_cols)} - {numeric_cols}
        - Categorical Variables: {len(categorical_cols)} - {categorical_cols}
        
        POTENTIAL STUDY VARIABLES IDENTIFIED:
        - Potential Outcomes: {potential_outcomes if potential_outcomes else 'None automatically identified'}
        - Potential Exposures/Treatments: {potential_exposures if potential_exposures else 'None automatically identified'}
        - Potential Time Variables: {potential_time_vars if potential_time_vars else 'None automatically identified'}
        
        DATA QUALITY ASSESSMENT:
        - Missing Values: {null_counts}
        - Sample Data Preview: {csv_preview.get('head', [])[:3]}
        
        STATISTICAL SUMMARY:
        {describe_stats}
        
        YOUR ROLE AS AN AI DATA SCIENTIST:
        1. **Data Understanding**: Automatically analyze the dataset structure and identify the type of study (observational, clinical trial, survey, etc.)
        2. **Intelligent Analysis**: Suggest appropriate statistical methods based on data types and research questions
        3. **Professional Communication**: Explain analyses in clear, professional language like a senior biostatistician
        4. **Comprehensive Testing**: Offer a full range of statistical tests including:
           - Descriptive statistics and data exploration
           - Hypothesis testing (t-tests, chi-square, ANOVA, etc.)
           - Regression analysis (linear, logistic, Cox proportional hazards)
           - Survival analysis (Kaplan-Meier, log-rank tests)
           - Advanced visualizations (forest plots, survival curves, etc.)
        5. **Result Interpretation**: Provide clinical/practical interpretation of statistical results
        6. **Visualization Recommendations**: Suggest appropriate plots and charts for different types of analyses
        
        IMPORTANT GUIDELINES:
        - Always examine the data structure first and identify what type of study this appears to be
        - When suggesting analyses, be specific about which variables to use and why
        - Always consider assumptions of statistical tests and suggest appropriate checks
        - Provide both statistical significance and clinical significance interpretations
        - Suggest appropriate visualizations for each type of analysis
        - Generate Python code when requested, using the full range of available libraries
        
        AVAILABLE LIBRARIES FOR ANALYSIS:
        pandas, numpy, scipy, statsmodels, matplotlib, seaborn, plotly, lifelines, sklearn, and more
        
        Please respond as a professional biostatistician would - with expertise, precision, and clear communication.
        """
        
        # Chat with Gemini using stable model
        chat = LlmChat(
            api_key=gemini_api_key,
            session_id=session_id,
            system_message=context
        ).with_model("gemini", "gemini-2.5-flash")
        
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
        error_msg = str(e)
        if "429" in error_msg or "Too Many Requests" in error_msg:
            raise HTTPException(
                status_code=429, 
                detail="Rate limit exceeded. Please wait a moment and try again. Consider using Gemini 2.5 Flash for faster response times."
            )
        elif "400" in error_msg or "Bad Request" in error_msg:
            raise HTTPException(
                status_code=400, 
                detail="Invalid API key or request. Please check your Gemini API key and try again."
            )
        else:
            raise HTTPException(status_code=500, detail=f"LLM Error: {error_msg}")

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
            'base64': base64,
            'go': go,
            'px': px,
            'ff': ff,
            'pio': pio,
            'sm': sm,
            'mcnemar': mcnemar,
            'KaplanMeierFitter': KaplanMeierFitter,
            'CoxPHFitter': CoxPHFitter,
            'logrank_test': logrank_test,
            'datetime': datetime,
            'json': json
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
                    plots.append({
                        'type': 'matplotlib',
                        'data': plot_data
                    })
                    buf.close()
                plt.close('all')
            
            # Handle Plotly plots (check for plotly figures in execution globals)
            plotly_plots = []
            for var_name, var_value in execution_globals.items():
                if hasattr(var_value, '_module') and 'plotly' in str(var_value._module):
                    try:
                        html_str = var_value.to_html(include_plotlyjs='cdn')
                        plotly_plots.append({
                            'type': 'plotly',
                            'html': html_str
                        })
                    except:
                        pass
            
            plots.extend(plotly_plots)
            
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
        
        # Enhanced analysis suggestions
        columns = csv_preview.get('columns', [])
        dtypes = csv_preview.get('dtypes', {})
        shape = csv_preview.get('shape', [0, 0])
        sample_data = csv_preview.get('head', [])
        
        # Analyze data structure
        numeric_cols = [col for col, dtype in dtypes.items() if 'int' in str(dtype) or 'float' in str(dtype)]
        categorical_cols = [col for col, dtype in dtypes.items() if 'object' in str(dtype)]
        
        context = f"""
        You are an Expert Biostatistician analyzing a medical research dataset.
        
        DATASET: {session['file_name']}
        STRUCTURE: {shape[0]} subjects, {shape[1]} variables
        
        VARIABLES IDENTIFIED:
        Numeric Variables ({len(numeric_cols)}): {numeric_cols}
        Categorical Variables ({len(categorical_cols)}): {categorical_cols}
        
        SAMPLE DATA:
        {sample_data[:5]}
        
        TASK: Provide 5-7 professional statistical analysis recommendations that would be appropriate for this dataset.
        
        For each analysis, provide:
        1. **Analysis Name**: Professional statistical test name
        2. **Purpose**: What research question it answers
        3. **Variables**: Specific columns to use (be precise)
        4. **Method**: Statistical approach (parametric/non-parametric)
        5. **Visualization**: Appropriate plot type
        6. **Clinical Relevance**: Why this analysis matters for medical research
        
        Consider these analysis categories:
        - Descriptive Statistics & Data Exploration
        - Group Comparisons (t-tests, ANOVA, chi-square)
        - Correlation & Regression Analysis
        - Survival Analysis (if time-to-event data present)
        - Multivariate Analysis
        - Advanced Visualizations (forest plots, survival curves)
        
        Format your response as a structured analysis plan that a biostatistician would create.
        Focus on clinically meaningful analyses that would be published in medical journals.
        """
        
        chat = LlmChat(
            api_key=gemini_api_key,
            session_id=f"{session_id}_suggestions",
            system_message="You are a statistical analysis expert. Provide suggestions in JSON format."
        ).with_model("gemini", "gemini-2.5-flash")
        
        response = await chat.send_message(UserMessage(text=context))
        
        return {"suggestions": response}
        
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "Too Many Requests" in error_msg:
            raise HTTPException(
                status_code=429, 
                detail="Rate limit exceeded. Please wait a moment and try again. Consider using Gemini 2.5 Flash for faster response times."
            )
        elif "400" in error_msg or "Bad Request" in error_msg:
            raise HTTPException(
                status_code=400, 
                detail="Invalid API key or request. Please check your Gemini API key and try again."
            )
        else:
            raise HTTPException(status_code=500, detail=f"LLM Error: {error_msg}")

@api_router.get("/sessions/{session_id}/analysis-history")
async def get_analysis_history(session_id: str):
    """Get analysis history for a session"""
    try:
        analyses = await db.analysis_results.find({"session_id": session_id}).sort("timestamp", -1).to_list(1000)
        return [AnalysisResult(**analysis) for analysis in analyses]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/sessions/{session_id}/save-analysis")
async def save_analysis_result(session_id: str, result: AnalysisResult):
    """Save analysis result to history"""
    try:
        result.session_id = session_id
        await db.analysis_results.insert_one(result.dict())
        return {"message": "Analysis saved successfully"}
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